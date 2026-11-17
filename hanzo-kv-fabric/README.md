# `hanzo-kv-fabric`

Epoch-fenced KV block ownership directory and framed TCP transport (`HKV1`) for disaggregating or replicating KV cache blocks across Hanzo Engine workers.

## Features

- **`EpochFencedKvDirectory`**: Monotonic per-worker epoch registration and stale-block fencing so restarted or preempted replicas never serve stale KV pages.
- **`KvBlockTransport`**: Framed `HKV1` async TCP push/pull transport with SHA-256 block integrity verification, epoch fencing, and a pluggable `KvStorageBackend` trait for host RAM or RDMA-pinned arenas.
- **`kv_block_hash`**: Deterministic 64-bit XXH3 content addressing over `(layer_group, token_ids)`.
- **`AdapterPagingRegistry`** (`adapter_paging` module): multi-tenant LoRA adapter paging per HIP-1345 §2.4 (S-LoRA, engine#96) — salted adapter keys so identical tenant adapters never share pages (§1.1), LRU eviction under a hard device-byte budget with explicit pinning, and fail-closed admission that reports `DeviceMemoryExhausted` instead of over-subscribing.
- **`CompositionPlanner`** (`composition` module): non-prefix RAG KV composition planning per HIP-1345 §2.4 (CacheBlend, engine#98) — plans reuse/recompute decisions for retrieved document chunks at shifting prompt positions, with fail-closed accounting for unknown, empty, ambiguous, and overflow cases.

## Example & Benchmark

```bash
cargo run --release -p hanzo-kv-fabric --example kv_fabric_bench
```
Multi-tenant LoRA adapter paging demo (engine#96 / S-LoRA):

```bash
cargo run --release -p hanzo-kv-fabric --example lora_paging_bench
```

Non-prefix RAG KV composition demo (engine#98 / CacheBlend):

```bash
cargo run --release -p hanzo-kv-fabric --example composition_bench
```

