"""Original-side DeepSearchQA parity runner (no Harbor involved).

Runs FrontierAgent's ReAct single agent (``stateful-react-agent``) on the
DeepSearchQA parity subset and grades every answer with FrontierAgent's own
DeepSearchQA judge (``benchmarks/public/judges/deepsearchqa.py``), which
implements the official Kaggle autorater prompt and the paper's P/R/F1.

Run it with FrontierAgent's interpreter, from the FrontierAgent checkout::

    cd /path/to/FrontierAgent        # this checkout
    export FRONTIER_AGENT_IMAGE=frontier-agent:0991a7a FRONTIER_AGENT_ENV_FILE=$PWD/.env
    .venv/bin/python deepsearchqa_parity/run_original.py \\
        --adapter-dir /path/to/adapters/src/deepsearchqa --out results/original-run1 --concurrency 4

Differences from FrontierAgent's stock runner (``run_subprocess.py``), each
needed for a like-for-like comparison:

1. **Question selection.** The 100 prompts come from the adapter's own
   ``--split parity`` selection (same CSV, seed and stratification), so both
   sides run the same prompts. FrontierAgent's runner can only filter by
   offset/limit/category.
2. **Agent execution.** Each question runs through the adapter's
   ``frontier_launcher.run_worker`` (a fresh container of FrontierAgent's own
   Docker image), the same launcher the Harbor-side agent uses. Inside, that is
   FrontierAgent's ``FrontierAgentAgent`` with ``_render_instruction(question)``,
   exactly as in ``run_one_task.py``. The judge runs on the host.
3. **Judge ``prompt_type`` fix.** FrontierAgent's ``score_answer()`` calls
   ``score_deepsearchqa()`` without ``prompt_type``, so every prompt is graded
   as "Single Answer". The official autorater receives the row's
   ``answer_type`` ("Single Answer" or "Set Answer"); this runner passes it.
4. **Metric.** F1 per prompt (the paper's primary metric) is recorded, not
   only FrontierAgent's binary fully-correct reward.
5. **Judge request parameters.** When ``DSQA_JUDGE_TEMPERATURE`` /
   ``DSQA_JUDGE_MAX_TOKENS`` / ``DSQA_JUDGE_REASONING_EFFORT`` are set, the
   judge sends exactly those (as the adapter's grade.py does) instead of
   FrontierAgent's defaults; see ``_install_parity_judge_params``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

AGENT_TIMEOUT_SEC = 3600  # same as the adapter's task.toml [agent].timeout_sec


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--adapter-dir", required=True, type=Path,
                   help="Path to adapters/src/deepsearchqa")
    p.add_argument("--out", required=True, type=Path, help="Run output directory")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--limit", type=int, default=None,
                   help="Only the first N parity prompts (sanity runs)")
    p.add_argument("--task-ids", nargs="+", default=None)
    p.add_argument("--source-csv", type=Path, default=None,
                   help="Local DSQA-full.csv (default: download, as the adapter does)")
    p.add_argument("--parity-size", type=int, default=100)
    p.add_argument("--parity-seed", type=int, default=42)
    p.add_argument("--regrade", action="store_true",
                   help="Do not run the agent: re-grade each trial's saved frontier_result.json "
                        "with the current judge settings (previous result kept as result.prev.json)")
    return p.parse_args()


def _load_parity_tasks(args: argparse.Namespace):
    sys.path.insert(0, str(args.adapter_dir.resolve() / "src"))
    from deepsearchqa.adapter import DeepSearchQAAdapter

    adapter = DeepSearchQAAdapter(
        output_dir=Path(tempfile.mkdtemp()),
        limit=args.limit,
        task_ids=args.task_ids,
        split="parity",
        parity_size=args.parity_size,
        parity_seed=args.parity_seed,
        source_csv=args.source_csv,
    )
    return adapter.selected_tasks()


def _install_parity_judge_params() -> dict:
    """Make FrontierAgent's DSQA judge send the same request as the adapter's grade.py.

    FrontierAgent's judge always sends ``max_completion_tokens=8192`` and drops
    ``reasoning_effort`` for non-OpenAI models, then retries with lower effort
    on empty output. For parity, both sides must send identical judge
    requests, so when ``DSQA_JUDGE_TEMPERATURE`` / ``DSQA_JUDGE_MAX_TOKENS`` /
    ``DSQA_JUDGE_REASONING_EFFORT`` are set (the same variables grade.py reads),
    they replace FrontierAgent's defaults, and the effort-downgrade cascade is
    replaced by a plain call. Retries on unparseable output are still handled
    by FrontierAgent's ``score_deepsearchqa_full`` loop. Prompt, parsing and
    P/R/F1 are untouched.
    """
    import logging

    import benchmarks.public.judges.deepsearchqa as dj
    from benchmarks.public.judges._common import _make_client

    params: dict = {}
    if v := os.environ.get("DSQA_JUDGE_TEMPERATURE", "").strip():
        params["temperature"] = float(v)
    if v := os.environ.get("DSQA_JUDGE_MAX_TOKENS", "").strip():
        params["max_tokens"] = int(v)
    if v := os.environ.get("DSQA_JUDGE_REASONING_EFFORT", "").strip():
        params["reasoning_effort"] = v
    if not params:
        return {}

    stock_build = dj._build_judge_kwargs
    log = logging.getLogger("run_original.judge")

    def build(prompt, **kwargs):
        built = stock_build(prompt, **kwargs)
        if "max_tokens" in params:
            built.pop("max_completion_tokens", None)
        built.update(params)
        return built

    async def call(kwargs, *, label):
        try:
            resp = await _make_client().chat.completions.create(**kwargs)
        except Exception as exc:  # retried by score_deepsearchqa_full
            log.warning("%s judge call error (%s): %s", label, kwargs.get("model"), exc)
            return ""
        return (resp.choices[0].message.content or "").strip()

    dj._build_judge_kwargs = build
    dj._judge_call_with_effort_fallback = call
    return params


async def _run_one(task, out_dir: Path, sem: asyncio.Semaphore, regrade: bool = False) -> dict:
    from benchmarks.public.judges.deepsearchqa import score_deepsearchqa_full
    from deepsearchqa.parity import frontier_launcher

    trial_dir = out_dir / "trials" / task.local_task_id
    result_path = trial_dir / "result.json"
    if regrade:
        saved = trial_dir / "frontier_result.json"
        prev = result_path if result_path.exists() else trial_dir / "failed_result.json"
        if prev.exists():
            prev.replace(trial_dir / "result.prev.json")
    elif result_path.exists():  # resume
        return json.loads(result_path.read_text(encoding="utf-8"))

    async with sem:
        t0 = time.monotonic()
        if regrade:
            prev_result = trial_dir / "result.prev.json"
            if saved.exists():
                agent = json.loads(saved.read_text(encoding="utf-8"))
            elif prev_result.exists():  # agent timeout: no worker result, empty answer
                old = json.loads(prev_result.read_text(encoding="utf-8"))
                agent = {"predicted": old.get("predicted_answer") or "", "error": old.get("agent_error")}
            else:
                raise RuntimeError(f"{task.local_task_id}: nothing to regrade in {trial_dir}")
        else:
            agent = await frontier_launcher.run_worker(task.problem, trial_dir, AGENT_TIMEOUT_SEC)
        predicted = agent.get("predicted") or ""
        agent_error = agent.get("error")

        # Same rule as the Harbor-side agent: no trajectory -> exception, not a score.
        exception = frontier_launcher.classify(agent)
        if exception:
            verdict, metrics = None, None
        else:
            verdict, metrics = await score_deepsearchqa_full(
                task.problem, task.answer, predicted, prompt_type=task.answer_type,
            )
            if metrics is None:
                exception = f"judge failed: verdict={verdict}"

    result = {
        "task_id": task.local_task_id,
        "answer_type": task.answer_type,
        "problem_category": task.problem_category,
        "gold_answer": task.answer,
        "predicted_answer": predicted,
        "agent_error": agent_error,
        "exception": exception,
        "verdict": verdict,
        "metrics": metrics,
        "f1": metrics["f1"] if metrics else None,
        "fully_correct": (metrics["category"] == "fully_correct") if metrics else None,
        "duration_seconds": round(time.monotonic() - t0, 2),
    }
    if exception is None:  # failed trials are retried on the next invocation
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        (trial_dir / "failed_result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{task.local_task_id}] f1={result['f1']} exception={exception}", flush=True)
    return result


async def _main(args: argparse.Namespace) -> None:
    root = Path.cwd()
    if not (root / "benchmarks" / "public").is_dir():
        sys.exit("Run from the FrontierAgent checkout (cwd must contain benchmarks/public)")
    sys.path.insert(0, str(root))
    from dotenv import load_dotenv

    load_dotenv(root / ".env")
    judge_params = _install_parity_judge_params()

    tasks = _load_parity_tasks(args)
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(tasks)} prompts -> {out_dir}", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(*(_run_one(t, out_dir, sem, args.regrade) for t in tasks))

    scored = [r for r in results if r["exception"] is None]
    summary = {
        "n_tasks": len(results),
        "n_scored": len(scored),
        "n_exceptions": len(results) - len(scored),
        "mean_f1": statistics.fmean(r["f1"] for r in scored) if scored else None,
        "fully_correct_rate": (
            statistics.fmean(1.0 if r["fully_correct"] else 0.0 for r in scored)
            if scored else None
        ),
        "agent_model": os.environ.get("OPENAI_MODEL"),
        "judge_model": os.environ.get("JUDGE_MODEL"),
        "judge_params": judge_params,
        "web_search_provider": os.environ.get("WEB_SEARCH_PROVIDER"),
        "frontier_agent_image": os.environ.get("FRONTIER_AGENT_IMAGE"),
        "regraded": args.regrade,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(_main(_parse_args()))
    sys.stdout.flush()
    os._exit(0)  # FrontierAgent judge/kernel threads can block interpreter shutdown
