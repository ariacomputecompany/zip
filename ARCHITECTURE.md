# zip Architecture

## Purpose

`zip` is the worker-side inference engine runtime. It owns execution state,
local scheduling inside a worker, checkpoint/KV handoff mechanics, and the
transport primitives needed to participate in distributed serving.

The design goal is to keep execution semantics explicit:

- explicit serving sessions
- explicit prefill and decode phases
- explicit backend boundaries
- explicit local decode batching
- explicit checkpoint/KV transfer state

## System Boundary

`zip` is not a control plane. It assumes some external authority can assign
work, return lease metadata, and persist global scheduler state.

`zip` owns:

- session-local runtime state
- provider selection and execution backend wiring
- batch formation inside a worker
- checkpoint serialization and import/export
- tensor-plane and collective transport
- local resource and memory management

`zip` does not own:

- global queueing across the cluster
- cross-worker placement policy
- persistent scheduler tables
- product shell concerns

## Core Modules

### `inference`

This is the runtime core.

- `coordinator.rs`
  Owns active sessions, drives phase transitions, forms decode batches, and
  advances execution.
- `backend.rs`
  Defines the stable execution backend boundary. The current baseline backend is
  Candle-based.
- `engine.rs`
  Holds serving-engine data structures such as session assignment, backend
  instance metadata, transfer policy, decode task, and decode batch plan.
- `fast_path.rs`
  Defines stable fast-path bucket planning, workspace requirements, metadata
  layout hashing, and graph/capture safety invariants.
- `job.rs`
  Defines request/result/progress types that move through the runtime.
- `stats.rs`
  Captures queue, batching, transport, recovery, and fast-path attribution
  telemetry.
- `artifact_loader.rs`
  Verifies and loads model artifacts.
- `tensor_ops.rs`
  Implements tensor-parallel math helpers used by the backend path.
- `forward_pass.rs`
  Contains the current backend-local execution path, including the paged live KV
  implementation used during active decode.
- `kv_cache.rs`
  Defines the transfer-side KV snapshot format plus the canonical live-KV
  metadata contract shared with recovery and handoff logic.

### `checkpoint`

This module is the durable handoff boundary.

- `types.rs`
  Defines checkpoint metadata, KV residency metadata, payload references, and
  checkpoint payload structure.
- `manager.rs`
  Implements checkpoint export/import, persistence, and handoff-friendly state
  management.

The checkpoint format is used for:

- same-worker recovery after local runtime loss
- cross-worker decode continuation after prefill handoff
- durable KV state exchange when live remote KV is not available

### `network`

This module owns worker-facing transport.

- `tensor_plane.rs`
  Dedicated tensor/data-plane transport, traffic classes, and metrics.
- `mesh_swarm.rs`
  Libp2p-based worker swarm wiring.
- `ring_gossip.rs`
  Ring topology logic and shard-neighbor reasoning.
- `ring_gossip_service.rs`
  Background service for gossip-driven topology updates.
- `tensor_message.rs`
  Tensor-plane wire message definitions.
- `events.rs`
  Network event types exposed to the runtime.

The network layer distinguishes latency-sensitive decode traffic from bulk
transfer traffic so the runtime can prioritize serving-critical work.

### `executor`

This module owns collective execution helpers.

- `ring_allreduce.rs`
  Ring collective implementation used by tensor-parallel execution.

### `model`

This module owns model and shard metadata.

- `registry.rs`
  Tracks model/shard inventory and assignments.
- `shard.rs`
  Describes shard metadata and status.

### `provider`

This module abstracts the local execution target.

- execution provider detection
- requested-provider validation
- selected-provider process-global initialization

The current provider model includes CPU, Metal, and CUDA targets.

### `device`, `discovery`, `connectivity`, `pki`

These modules support runtime participation in a serving network:

- device config and capability reporting
- multicast and peer discovery
- direct reachability and connectivity state
- key material, node identity, and certificate utilities

## Session Model

The runtime centers on explicit serving sessions.

An active session holds:

- session identity
- model/shard assignment
- execution phase
- backend state
- queue/accounting state
- KV position and checkpoint metadata

This differs from an opaque “whole job” runtime. Prefill and decode are
different runtime states, and the runtime can retain or reconstruct a session
across that boundary.

## Runtime Architecture Direction

The runtime now has a sharper split between:

- local executor concerns
- live KV layout concerns
- transfer/recovery concerns
- serving transport concerns
- control-plane-facing progress and lease concerns

That split matters because `zip` is intentionally not a generic single-node
engine. It is a distributed worker runtime. The architecture therefore keeps
session continuity, ownership, handoff, regroup, and recovery semantics
explicit, while still pushing the local decode path toward a more stable
fast-path execution model.

## Execution Flow

### 1. Work materialization

External control logic provides assignment metadata. `zip` converts that into a
local `EngineSessionState` and backend instance.

### 2. Prefill

Prefill loads prompt state, runs the first forward pass, and can produce:

- first-token output
- incremental KV state
- checkpoint exportable handoff state

### 3. Handoff boundary

If execution moves or local runtime state must survive process loss, the runtime
serializes checkpoint state through the checkpoint manager.

### 4. Decode queue admission

Decode work is tracked explicitly. The coordinator admits ready sessions into a
local decode queue and evaluates batch pressure, resource pressure, and runtime
policy before stepping.

### 5. Decode batch formation

The coordinator builds a `DecodeBatchPlan` from queued sessions. The plan can
include:

- target session count
- target batch size
- admitted KV footprint
- deferred sessions

### 6. Decode step execution

The backend executes one decode step for the admitted sessions. Session-local KV
and output state are updated, and finished sessions are retired or checkpointed.

## Fast-Path Execution Model

The worker runtime now distinguishes between:

- fallback execution
- accelerated fast-path execution

The fast path is described through explicit contracts in `engine.rs` and
planned through `fast_path.rs`.

The important properties are:

- explicit prefill and decode phase plans
- provider-specific execution profiles
- stable decode buckets
- stable prefill buckets
- workspace reservation requirements
- metadata layout hashing
- graph/capture safety validation

This keeps the coordinator from rebuilding execution shape ad hoc every step.
Instead, decode and prefill can be admitted against a known bucket/workspace
contract and validated before execution.

In the current runtime, the accelerated decode path is intentionally narrower
than the generic fallback path. Fast-path decode requires:

- homogeneous accelerated providers across the admitted microbatch
- a backend contract that advertises decode-microbatch support
- a KV runtime contract that requires paged live KV and append-only decode
- a bucket/workspace plan that validates against the live session metadata

That narrow contract is deliberate. It keeps the hot path explicit and
measurable instead of silently widening into a “best effort” execution mode.

## Backend Boundary

The `ExecutionBackend` trait is the provider-facing execution boundary.

That boundary is responsible for:

- materializing backend-local runtime state
- running prefill and decode operations
- translating model tensors into execution outputs
- supporting incremental decode against preserved KV state

The baseline backend uses Candle. The rest of the runtime is written so a
different provider-specialized backend can be installed without rewriting
session coordination logic.

The backend boundary now also carries:

- provider kind
- optimization profile
- local executor contract
- live KV export/import support
- fast-path planning context

That makes the worker runtime able to decide whether a decode batch is eligible
for the accelerated path without mixing provider-specific logic back into the
coordinator.

## Live KV Versus Transfer KV

One of the important architecture changes is the explicit separation between:

- live execution KV
- transfer/checkpoint KV

Live execution KV is now treated as a paged, block-table-oriented runtime
structure shaped around active decode needs.

Transfer/checkpoint KV is treated as a durable/exportable representation shaped
around:

- recovery
- cross-worker handoff
- checkpoint persistence

`kv_cache.rs`, `forward_pass.rs`, and `checkpoint/types.rs` define the seam
between those two worlds through:

- live KV layout metadata
- live sequence metadata
- transfer hooks
- recovery points
- checkpoint handoff descriptors

This is what keeps recovery semantics from dictating the hot in-memory decode
layout.

## Local Scheduling Model

`zip` is not the global scheduler, but it does perform local scheduling inside a
worker.

The coordinator tracks:

- active sessions
- queued decode tasks
- fairness state
- batch pressure
- KV-budget deferrals
- batch-target overrides provided by external scheduling authority

This local scheduling layer is what lets a worker turn explicit session work
into practical microbatched decode execution.

The current local scheduling contract also preserves:

- fairness ordering through monotonic decode epochs
- scheduler-owned batch targets
- KV-budget deferrals
- batch-capacity deferrals
- multi-session decode admission

That allows a worker to honor external scheduling intent while still making
worker-local batching decisions cheaply.

## Resource Management

`resource_manager.rs` owns local memory accounting and platform-specific memory
locking helpers.

This layer is important because the runtime needs to reason about:

- total system memory
- available locked memory
- per-session KV growth
- decode admission under memory pressure

## Transport and Collective Path

Distributed execution is split into two cooperating layers:

- tensor-plane transport for point-to-point serving traffic
- collective execution helpers for ring-based tensor-parallel work

This keeps serving-critical transport concerns separate from collective math and
lets the runtime choose different traffic priorities for different classes of
work.

The transport path now has a more explicit serving-lane model:

- reduce-scatter lane
- all-gather lane
- control lane
- bulk-transfer lane
- checkpoint lane

The collective layer can describe whether a given lane is:

- a blocking stage boundary
- step-pipelined
- safe for background overlap with decode work

That gives the worker runtime a way to overlap checkpoint/bulk transfers with
decode where safe, while still draining those background transfers before
blocking collective boundaries when correctness requires it.

## Control-Plane Interaction Model

`zip` does not own the control plane, but it does deliberately shape how often a
worker talks to one.

The active decode path now uses a coarser interaction model:

- queue observation is throttled during idle claim loops
- decode lease renewal is interval-based rather than token-based
- decode progress is buffered and coalesced before being reported
- batch telemetry is aggregated into session and event surfaces
- final result and lease release still happen at explicit completion/failure
  boundaries

The design goal is to preserve:

- queue visibility
- operator introspection
- fairness and cohort semantics
- ownership correctness

without letting token-by-token decode cost be dominated by control-plane churn.

## Measurement and Attribution

`stats.rs` is no longer just a generic success/failure counter. It is part of
the runtime architecture because it provides the attribution layer needed to
understand whether a slowdown came from:

- local executor behavior
- batching/admission behavior
- collective/transport behavior
- checkpoint/recovery behavior

The runtime now records surfaces such as:

- decode microbatch counts
- batch-size and KV-footprint telemetry
- fast-path plan rates
- multi-session batch rates
- deferred-session counts per microbatch
- arena reuse
- transport bytes and wait times
- collective wait-share metrics
- collective transport share of runtime
- checkpoint and recovery counters
- recovery success and rejection rates

This keeps the worker runtime observable enough to support real benchmark and
regression analysis instead of only pass/fail testing.

## Client Protocol Surface

`src/api` contains client-side protocol types and registration/claim helpers
used by workers to talk to an external control plane. Those types stay in this
repo because they are part of the worker runtime contract, even though the
server-side implementation is intentionally out of scope here.

## Operational Model

The public library is meant to be embedded by worker processes or higher-level
wrappers. A typical integration provides:

- its own work-assignment source
- its own persistence authority
- its own operator APIs
- its own product shell

`zip` provides the runtime and transport substrate underneath that integration.
