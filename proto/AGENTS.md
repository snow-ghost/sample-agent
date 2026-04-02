# AGENTS

This subtree contains source protocol definitions. Treat it as the contract boundary for generated SDKs and runtimes.

## Local Goals

- Keep schemas general, capability-based, and minimally opinionated about any one benchmark.
- Prefer names and fields that describe the runtime contract, not one sample agent's implementation.
- Preserve backward compatibility whenever possible.

## Schema Rules

- Add fields instead of reinterpreting existing fields.
- Keep zero values meaningful and safe.
- Document normalization, path semantics, line numbering, limits, and result shapes in comments.
- Prefer explicit enums and typed messages over overloaded strings.
- Make terminal outcomes explicit in the protocol if the runtime needs to record them.
- Keep request/response shapes symmetrical where it improves predictability.

## Change Discipline

- When adding a capability, first ask whether it belongs in the protocol or only in sample-agent policy.
- Avoid embedding prompt logic, benchmark-specific policy, or model assumptions in proto comments or field names.
- Any schema change should be reviewed for generated SDK impact on both `sandbox-py` and `pac1-py`.
