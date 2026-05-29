# zip

`zip` is a distributed inference engine runtime for explicit serving sessions,
prefill/decode execution, checkpoint-backed KV handoff, and tensor-parallel
worker coordination.

We built our own inference engine because Mesh needs explicit serving sessions,
prefill/decode separation, checkpoint-backed KV handoff, and distributed
tensor-parallel execution as a first-class worker boundary. 

The goal is higher usable throughput, lower per-worker memory pressure, and
better steady-state `tok/s` for sharded models across multiple workers.

In the larger Mesh project, `zip` is the worker-side execution runtime beneath the
control plane, scheduler, and product surfaces.

## Scope

`zip` currently provides:

- runtime coordination for inference sessions
- provider-aware execution backends
- explicit prefill and decode phases
- microbatch-oriented decode stepping
- fast-path decode planning with stable bucket/workspace contracts
- provider-aware decode admission guardrails for batch cohesion
- checkpoint export/import and KV handoff metadata
- worker-to-worker networking and tensor-plane transport
- lane-aware serving transport with stable stream binding
- ring collective execution primitives, including optimized 2-worker decode
  all-reduce and direct-byte larger-ring transport paths
- collective and batching attribution surfaces for production benchmarking
- device capability, connectivity, and identity helpers
- model shard metadata and artifact loading

## Repository Layout

- `src/inference`
  Runtime coordinator, execution backend boundary, job/session state, batch
  planning, tensor operations, and artifact loading.
- `src/checkpoint`
  Checkpoint formats, KV residency metadata, and checkpoint manager logic.
- `src/network`
  Tensor-plane transport, mesh swarm wiring, gossip, and transport metrics.
- `src/executor`
  Collective execution primitives, including ring all-reduce.
- `src/model`
  Model, shard, and registry metadata.
- `src/provider`
  Execution provider detection and selection.
- `src/device`, `src/discovery`, `src/connectivity`, `src/pki`
  Device identity, discovery, reachability, and PKI helpers used by the runtime.
- `src/api`
  Client-side control-plane protocol types used by workers.
- `src/telemetry`, `src/observability`
  Event publication and logging utilities.

## Public API

The crate root re-exports the main runtime surfaces:

- `InferenceCoordinator`
- `InferenceConfig`
- `InferenceRequest`
- `InferenceResult`
- `ExecutionProviderKind`
- `CheckpointManager`
- `TensorPlane`
- `WorkerRing`
- `ResourceManager`

## Architecture Summary

`zip` models inference as explicit serving sessions with persistent runtime
state across prefill, decode, handoff, and recovery boundaries.

Each active session owns:

- stable session identity
- phase-specific execution state
- backend-specific runtime state
- shard and provider metadata
- checkpoint/KV handoff metadata

The runtime executes prefill and decode as separate phases. Prefill can emit a
checkpoint-backed handoff boundary. Decode work is admitted through an explicit
runtime queue and advanced with batch-oriented stepping. Accelerated decode is
planned against explicit backend contracts, bucket ceilings, workspace limits,
and live-KV invariants.

## Runtime Highlights

- explicit session lifecycle with checkpoint-backed handoff and resume
- decode microbatch formation with fairness, KV-budget, and fast-path cohesion
  guardrails
- stable serving lanes for reduce-scatter, all-gather, control, bulk, and
  checkpoint traffic
- logical collective stream IDs bound to durable underlying peer channels
- direct-byte collective transport for matrix and tensor all-reduce
- sender-position-aware slot mailboxes for overlapped larger-ring receive paths
- provider-specialized execution paths across CPU, Metal, and CUDA-capable
  deployments
- pooled Metal collective scratch for lower allocation churn during decode
- runtime telemetry for collective wait share, transport cost, batch deferrals,
  and recovery behavior

See [ARCHITECTURE.md](/Users/deepsaint/Desktop/zip/ARCHITECTURE.md) for the
detailed design.

## Build

```bash
cargo fmt --check
cargo check
cargo test
```

## Example

```rust
use zip::{default_execution_provider, detect_execution_providers, InferenceConfig};

let providers = detect_execution_providers();
let provider = default_execution_provider(&providers);
let _config = InferenceConfig::default();

assert!(providers.iter().any(|candidate| candidate.kind == provider));
```

## Open Source Notes

See [OPEN_SOURCE.md](/Users/deepsaint/Desktop/zip/OPEN_SOURCE.md).

## License

MIT
