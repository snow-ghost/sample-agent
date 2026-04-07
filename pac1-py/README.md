# BitGN PAC1 Python Sample

Runnable Python sample for the `bitgn/pac1-dev` benchmark.

This sample follows the same control-plane flow as the sandbox demo, but it targets the PCM runtime (`bitgn.vm.pcm`) instead of the mini sandbox VM.

You will need to provide your own `OPENAI_API_KEY` (or swap the OpenAI client for a provider of your choice).

Quick start:

1. Export `OPENAI_API_KEY`
2. Optionally export `OPENAI_BASE_URL`, `BENCHMARK_HOST`, `BENCHMARK_ID`, or `MODEL_ID`
3. Run `make sync`
4. Run `make run`

Run it with:

```bash
uv run python main.py
```

Useful environment overrides:

- `OPENAI_BASE_URL` points the OpenAI client to an OpenAI-compatible provider
- `BENCHMARK_HOST` defaults to `https://api.bitgn.com`
- `BENCHMARK_ID` defaults to `bitgn/pac1-dev`
- `MODEL_ID` defaults to `gpt-4.1-2025-04-14`

Generalization and scorer-blind development docs:

- `docs/agent-generalization-plan.md`
- `docs/future-benchmark-signals.md`
- `docs/runtime-log-analysis.md`
- `docs/failure-taxonomy.md`
- `evals/scorer-blind-corpus.md`
