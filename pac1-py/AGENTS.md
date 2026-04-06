# AGENTS

This subtree contains the PAC1 Python contest agent. Optimize for correctness, small surface area, and low dependency count.

## Local Goals

- Keep the implementation minimal and runnable with the packages already present in `pyproject.toml`.
- Reuse the generated BitGN SDK and protobuf types directly instead of wrapping everything in another framework.
- Avoid benchmark-id branching or task-id hardcoding unless there is no protocol-level alternative.
- Prefer generic file-system reasoning over PAC1-specific heuristics.
- Prefer capability discovery plus broad intent classes over phrase-specific routing. Narrow keyword heuristics are a last-resort fallback, not the primary architecture.

## Preferred Structure

- `main.py`: benchmark orchestration only
- `agent.py`: agent loop, typed tool schemas, formatting, and runtime dispatch

If `agent.py` grows further, split by responsibility:

- `config.py`: env parsing and defaults
- `models.py`: request/response schemas for the LLM contract
- `runtime.py`: PCM dispatch and response formatting
- `loop.py`: planning/execution loop

Do not split earlier than necessary. Small and direct is still the default.

## Implementation Rules

- Keep the loop deterministic and bounded.
- Bootstrap with explicit grounding calls before the free-form task loop.
- Preserve a generic command vocabulary based on runtime capabilities: tree, list, read, search, write, move, delete, mkdir, context, answer.
- Format tool outputs in stable shell-like shapes when that improves model grounding.
- Keep completion reporting explicit and typed.
- Put tunables behind environment variables or small constants, not string literals scattered through the code.
- Prefer extraction of tiny pure helpers over repeated inline formatting logic.
- Use BDD for policy and loop changes. Express scenarios as `Given / When / Then`, then implement deterministic tests for those scenarios with minimal tooling.
- Treat contest paths as case-sensitive. Reuse the exact casing returned by the runtime and try known filename variants only when the policy intentionally supports them.

## Dependency Rules

- Standard library first.
- Keep `pydantic` only for typed LLM I/O contracts unless another use is clearly justified.
- Do not introduce agent orchestration frameworks when a direct loop is shorter and clearer.
- Do not add persistence, queues, or databases for this sample.

## Pattern Guidance

- Default runtime architecture: bounded `ReAct` loop with a typed stop action.
- Add a lightweight verify step before reporting success if the task mutated files.
- Use reflection only when it materially improves correctness; otherwise prefer another tool call.
- `Ralph`-style repeated attempts are acceptable only as a bounded outer loop with explicit stopping conditions.
- Do not implement self-modifying or `Darwin Gödel Machine` style runtime behavior inside the contest agent.
