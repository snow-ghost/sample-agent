# AGENTS

## Priority Model

When instructions conflict, use this order:

1. System instructions
2. Developer instructions
3. User request
4. This root `AGENTS.md`
5. The most specific `AGENTS.md` inside the subtree being changed

More specific `AGENTS.md` files refine the local implementation details for their subtree. They must not contradict higher-priority instructions.

## Repository Map

- `proto/`: source-of-truth protocol definitions for control-plane and runtime exchange
- `pac1-py/`: Python sample agent for the PAC1 contest/runtime
- `sandbox-py/`: smaller demo agent against the mini runtime

## Global Engineering Goals

- Keep the codebase small, readable, and runnable with minimal dependencies.
- Prefer reuse over duplication. If a concept already exists in the repo, extend or extract it instead of cloning logic.
- Avoid hardcoding benchmark-specific assumptions unless the protocol or runtime explicitly requires them.
- Design for capability-driven behavior, not for one task catalog.
- Keep prompts, policies, limits, and model ids configurable through environment variables or small typed config objects.
- Favor stable interfaces and narrow adapters around external SDKs.

## Architectural Defaults For Agents

- Separate `protocol`, `runtime adapter`, `agent policy`, and `entrypoint`.
- Keep the agent loop explicit. A good default is:
  `observe -> update state -> choose next action -> execute tool -> record observation -> stop or repeat`.
- Use typed request/response models at boundaries. Inside the loop, keep state compact and serializable.
- Treat prompt text as policy configuration, not as hidden business logic.
- Encode terminal outcomes explicitly. Do not rely on implicit success/failure states.
- Make side effects go through one adapter layer so they can be audited, tested, and replaced.
- Prefer a small strategy interface over branching on benchmark/task ids.

## Recommended Agent Patterns

- `ReAct` loop: best default for bounded tool-use tasks. Alternate between reasoning and action, but keep reasoning short and grounded in tool output.
- `Plan -> Execute -> Verify`: use when tasks are longer than a few tool calls. The plan can be compact; verification should be explicit.
- `Supervisor / worker`: use only when there are truly independent subtasks. Avoid it in minimal contest agents unless parallelism clearly helps.
- `State machine`: use for deterministic lifecycle steps such as bootstrap, grounding, execution, completion, and failure handling.
- `Reflection / critique`: keep this lightweight. One verifier pass is usually enough; avoid infinite self-critique loops.
- `Ralph loop`: useful as an outer retry shell around a compact inner loop. Good for "attempt -> inspect result -> continue" workflows, but cap attempts and preserve deterministic logs.
- `Darwin Gödel Machine` style self-improvement: treat as an offline research or benchmarking pattern, not as the default contest runtime loop. Self-modification can improve scaffolding, but it adds evaluation cost, safety risk, and reproducibility problems.

Pragmatic default for this repository:

- Inner loop: typed `ReAct` with tool calls
- Outer loop: bounded `plan/execute/verify`
- Evolutionary or self-modifying loops: only offline, never as the first-line runtime design

## Python Best Practices

- Prefer the standard library first.
- Add a third-party dependency only when it removes substantial complexity or is already required by the SDK/runtime.
- Keep modules focused. A single file is fine for a tiny sample, but split once protocol models, formatting, transport, and policy start mixing.
- Use type hints on public functions and module-level constants for configuration defaults.
- Keep I/O at the edges. Pure helpers should accept data and return data.
- Use small dataclasses or pydantic models for typed boundaries, not for every internal value.
- Avoid global mutable state except for constant configuration caches.
- Keep error handling explicit and user-visible. Surface transport failures, validation failures, and policy failures differently.
- Prefer deterministic formatting and deterministic ordering in logs.
- Write code so it can run under a tight environment with only `uv`, the SDK packages, and the model client.

## Testing And Validation

- Prefer fast local validation over heavy test scaffolding.
- For protocol changes, check backward compatibility and generated SDK impact.
- For agent changes, validate the bootstrap path, tool dispatch, failure path, and completion path.
- Test the generic loop behavior before task-specific heuristics.

## Change Discipline

- Before adding new abstractions, check whether the same result can be achieved by extracting a helper or config object.
- Before adding a new dependency, document why stdlib or existing packages are insufficient.
- If a change is specific to one benchmark, isolate it behind a named policy or adapter.
