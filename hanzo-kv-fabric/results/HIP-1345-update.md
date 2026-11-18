# HIP-1345 Status Update — Multi-Tenant LoRA Paging (Bounty #15, PR #104)

This is the proposed status text for HIP-1345 §2.4 (Model-Loading IO Templates &
Multi-Tenant LoRA Paging, Bounties #14–#15), per the live HIP text at
`hanzoai/hips/HIPs/hip-1345-agent-aware-model-serving-and-cache-reuse.md`
(§2.4 covers Bounties #14–#15; the earlier §2.3 reference in this PR's first
review cycle was a mislabel — §2.3 is Bounties #13/#18, KV-residency scheduling).

## Proposed §2.4 addendum (status note)

> **§2.4 Status — Multi-Tenant LoRA Paging (Bounty #15).** A fabric-level
> paging *policy* layer is implemented in `hanzo-kv-fabric`
> (`adapter_paging` module): a device-byte-budgeted `AdapterPagingRegistry`
> with LRU eviction, explicit per-tenant pinning, tenant-salted adapter
> identity (HIP-1345 §1.1), and stats-accounted hit/miss/eviction counters.
> Under a deterministic standing eviction storm (8 tenants × 2 adapters ×
> 4 layers × 4 MiB pages = 256 MiB demand against a 40 MiB budget, 32
> rounds), the unpinned-LRU baseline achieves 0.0% hit ratio (0/2048) with
> 2038 evictions; pinning one tenant's adapters lifts that tenant to
> 96.9% (248/256) at zero additional device budget, with its 8/8 pages
> surviving the storm and policy cost ~302 ns (resident hit) / ~709 ns
> (admission) per lookup. Raw numbers: `results/lora-paging-results.json`
> in the PR. These are fabric-policy-layer figures on the authoring host —
> upstream S-LoRA device figures (adapters per GPU, adapter-switch
> latency) remain the original authors' own and are NOT treated as Hanzo
> expectations until reproduced on Hanzo hardware.

## What this PR proves

- The §1.1 salt-isolation invariant holds at the paging tier: a foreign salt
  cannot resolve a registered adapter (`UnknownAdapter`, asserted in the bench).
- Tenant pinning is sufficient to make a tenant's hot pages eviction-immune
  under a standing storm, without new memory budget.
- The policy tier is sub-microsecond per lookup (302/709 ns), so it is not the
  bottleneck when the engine's device-side paging lands.

## Explicit out-of-scope statement

This PR does **not** cover:

1. **Adapter training/fine-tuning lifecycle** — Cloud owns lifecycle & billing;
   this PR only pages already-registered adapter layers.
2. **Runtime hot-swap into the attention path** — materializing paged adapter
   layers as device-resident LoRA weights is the engine's job (Phase 2).
3. **Device memory management / device-side I/O** — no GPU is simulated; the
   registry is a byte-budgeted policy structure over logical pages.
4. **Cross-worker replication of paged adapters** — a worker-local registry;
   distributed directory replication is fabric-tier future work.
5. **Multi-GPU placement / NUMA-aware page ownership.**
6. **Pin-budget negotiation between tenants** — the bench pins one demo tenant;
   a quota arbiter (admission control for pin requests) is future work.
7. **Bounty #14 (weight-loading IO templates)** — separate deliverable.

## Follow-ups for §2.4 completion

- Hanzo-hardware run (device-resident paged attention path) to produce the
  Hanzo-expectation numbers the contract requires: adapters per GPU,
  adapter-switch latency, mixed-rank throughput.
- Pin-quota arbiter so pinning cannot starve unpinned tenants (fairness bound).
