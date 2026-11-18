# `hanzo-kv-fabric`

Epoch-fenced KV block ownership directory and framed TCP transport (`HKV1`) for disaggregating or replicating KV cache blocks across Hanzo Engine workers.

## Features

- **`EpochFencedKvDirectory`**: Monotonic per-worker epoch registration and stale-block fencing so restarted or preempted replicas never serve stale KV pages.
- **`KvBlockTransport`**: Framed `HKV1` async TCP push/pull transport with SHA-256 block integrity verification, epoch fencing, and a pluggable `KvStorageBackend` trait for host RAM or RDMA-pinned arenas.
- **`kv_block_hash`**: Deterministic 64-bit XXH3 content addressing over `(layer_group, token_ids)`.

## Example & Benchmark

```bash
cargo run --release -p hanzo-kv-fabric --example kv_fabric_bench
```
- **`ResidencyScheduler`** (`residency_scheduler` module): KV-residency-aware request scheduling per HIP-1345 §2.3 (Bidaw, engine#94) — separates multi-turn requests by KV residency tier (device → host → SSD), reorders smallest-KV-first within each tier, and simulates TTFT/goodput against a FIFO baseline on a deterministic mixed warm/cold workload. Mechanism 2 (response-predicted eviction) is documented as out-of-scope.
