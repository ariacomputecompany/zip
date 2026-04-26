# Open Source Notes

## Repository Scope

This repository publishes the `zip` worker-side inference engine runtime.

Included here:

- runtime coordination
- execution backend boundary
- checkpoint and KV handoff logic
- transport and collective execution primitives
- model/shard metadata
- worker protocol client types

Intentionally not included here:

- control-plane server implementation
- product UI
- billing or accounting systems
- operator-specific deployment wrappers

## Packaging Rules

- Keep the crate library-first.
- Keep public exports centered on runtime and engine concepts.
- Avoid product-shell abstractions in the public API.
- Treat control-plane interaction as a protocol/client boundary, not as an
  embedded server dependency.

## Documentation Rules

- Document the current implementation only.
- Describe missing pieces as out of scope, not as implied features.
- Keep terminology consistent with the code: sessions, prefill, decode,
  checkpoints, KV, tensor plane, collective execution.

## Security and Hygiene

- Do not commit credentials, access tokens, or environment-specific secrets.
- Do not hard-code internal service endpoints.
- Keep examples local and deterministic.
- Prefer repository-neutral names in public docs and examples.

## Verification

Expected local verification for library changes:

```bash
cargo check
cargo test
```

Runtime and system-level verification lives in the main deployment repository
that integrates this engine with its scheduler and control plane.
