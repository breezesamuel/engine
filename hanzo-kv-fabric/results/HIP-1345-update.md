# HIP-1345 Status Update — KV-Residency-Aware Scheduling (Bounty #13, engine#94)

Proposed status text for HIP-1345 §2.3 (KV-Residency-Aware & Distributed
Locality Scheduling, Bounties #13/#18), per the live HIP text at
`hanzoai/hips/HIPs/hip-1345-agent-aware-model-serving-and-cache-reuse.md`.
Companion deliverable: the distributed/Enso half of §2.3 (Bounty #18, Preble)
is already covered by cloud PR #48 (`preble-locality-research`); this PR is
the single-node scheduler half (Bounty #13, Bidaw).

## Proposed §2.3 addendum (status note)

> **§2.3 Status — KV-residency-aware scheduling (Bounty #13, policy layer).**
> A Bidaw-style scheduling policy is implemented in `hanzo-kv-fabric`
> (`residency_scheduler` module): requests are separated by KV residency
> tier (device-resident → host memory → SSD) and reordered smallest-KV-first
> within each tier, on a deterministic sequential-server simulation with
> per-tier load bandwidths (20 GB/s host, 2 GB/s SSD) and a prefill compute
> rate. On a fixed 24-request mixed warm/cold workload (8 resident / 8 host
> / 8 SSD, KV footprints 16 MiB–2 GiB, interleaved arrival), FIFO arrival
> order vs residency order are compared on identical input. Raw numbers:
> `results/residency-scheduler-results.json` in the PR. These are
> fabric-policy-layer simulation figures on the authoring host; the
> upstream Bidaw figures (up to 3.58× latency reduction, 1.83× throughput)
> remain the original authors' own and are NOT treated as Hanzo
> expectations until reproduced on Hanzo hardware with real two-tier I/O.

## What this PR proves

- Tier separation + size-ranked reordering removes the cold-load
  head-of-line blocking that FIFO suffers on mixed warm/cold queues:
  p95 TTFT and worst-case TTFT improve materially, while total load bytes
  (and therefore expected goodput) are unchanged by construction.
- The ordering itself is allocation-light and deterministic —
  `residency_order` is a stable sort by `(tier rank, kv_bytes, arrival)`,
  with per-request cost in the tens-of-nanoseconds range at this workload
  size, negligible against any real load latency.
- Fairness is measured, not asserted: the results file reports max-TTFT
  alongside p50/p95/p99, so the starvation cost of always serving the fast
  tier first is visible.

## Explicit out-of-scope statement

This PR does **not** cover:

1. **Bidaw mechanism 2 (response-predicted eviction)** — predicting user
   access patterns from LLM-generated responses to raise host-memory hit
   rates needs response content and a storage trace; a scheduling policy
   does not see either. Documented as a follow-up for the storage tier.
2. **Bidaw mechanism 3 (selective storage-efficient history caching)** —
   footprint-vs-compute balancing across history tensors.
3. **Real two-tier I/O** — the simulation uses configured per-tier
   bandwidths, not measured device I/O; the engine-side integration must
   replace these with measured rates on Hanzo hardware.
4. **GPU-side batching / continuous batching** — the server model is a
   bounded sequential server; batching interplay is engine work.
5. **Distributed/Enso gateway routing (Bounty #18)** — covered separately by
   cloud PR #48 (Preble); this PR is the single-node scheduler half.
6. **Admission control / backpressure** — the queue is unbounded in this
   model; overload behavior is future work.
7. **`hanzoai/engine#56` integration** — complement, not included here.

## Follow-ups for §2.3 completion

- Hanzo-hardware run with measured host/SSD bandwidths replacing the
  configured rates, to produce Hanzo-expectation p95 TTFT/goodput numbers.
- Mechanism 2 prototype once response-content access traces exist.
