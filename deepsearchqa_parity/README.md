# DeepSearchQA parity: original side

This branch adds one file to FrontierAgent at commit `0991a7a9`: `run_original.py`, the original-side runner for the
[Harbor DeepSearchQA adapter](https://github.com/harbor-framework/adapters/tree/main/src/deepsearchqa) parity
experiment. No FrontierAgent code is modified.

It runs FrontierAgent's ReAct single agent (`stateful-react-agent`) on the adapter's 100-prompt parity subset and grades
each answer with FrontierAgent's own DeepSearchQA judge (`benchmarks/public/judges/deepsearchqa.py`, official Kaggle
autorater prompt, per-prompt P/R/F1). The Harbor side runs the same agent through the adapter's `DeepSearchQAFrontierAgent`.

## What differs from FrontierAgent's stock runner, and why

1. **Question selection.** The prompts come from the adapter's `--split parity` selection (stratified by answer type,
   seed 42), so both sides run the same 100 prompts.
2. **Agent execution.** Each question runs in a fresh container of FrontierAgent's Docker image
   (`SANDBOX_BACKEND=container`), started by the adapter's `frontier_launcher.run_worker`, the same launcher the Harbor
   side uses. Inside the container: `FrontierAgentAgent` with `_render_instruction(problem)`, as in `run_one_task.py`.
3. **Judge `prompt_type` fix.** `score_answer()` calls `score_deepsearchqa()` without `prompt_type`, so every prompt is
   graded as "Single Answer". The runner passes each row's `answer_type`, as the official autorater does.
4. **Metric.** Per-prompt F1 (the paper's primary metric) is recorded, plus the fully-correct rate.
5. **Judge request parameters.** When `DSQA_JUDGE_TEMPERATURE` / `DSQA_JUDGE_MAX_TOKENS` / `DSQA_JUDGE_REASONING_EFFORT`
   are set, the judge sends exactly those (as the adapter's `grade.py` does) and makes one plain call per attempt instead
   of the reasoning-effort downgrade cascade. Prompt, parsing and scoring are untouched.

Failure rules (same as the Harbor side): agent timeout (3600 s) scores F1 0; a worker crash or a FrontierAgent pipeline
error is an infrastructure exception and is not scored; a judge failure is retried on the next invocation
(`--regrade` re-grades saved answers without re-running the agent).

## Run

```bash
docker build -t frontier-agent:0991a7a .
uv sync --extra eval --extra sandbox --extra document-readers   # see FrontierAgent README
cp .env.example .env                      # then fill in the keys below
export FRONTIER_AGENT_IMAGE=frontier-agent:0991a7a FRONTIER_AGENT_ENV_FILE=$PWD/.env
.venv/bin/python deepsearchqa_parity/run_original.py \
    --adapter-dir /path/to/adapters/src/deepsearchqa --out results/original-run1 --concurrency 4
```

`.env` used for the recorded runs (keys omitted):

```
OPENAI_BASE_URL=https://api.aiand.com/v1
OPENAI_MODEL=deepseek-ai/deepseek-v4.1-flash-1017
JUDGE_BASE_URL=https://api.aiand.com/v1
JUDGE_MODEL=zai-org/glm-5.3-flash
WEB_SEARCH_PROVIDER=parallel
JINA_BASE_URL=https://r.jina.ai
DSQA_JUDGE_TEMPERATURE=1
DSQA_JUDGE_MAX_TOKENS=32768
DSQA_JUDGE_REASONING_EFFORT=max
```

Each run writes `trials/<task-id>/result.json` and `summary.json` (`mean_f1`, `fully_correct_rate`). Results are in the
adapter's `parity_experiment.json`.
