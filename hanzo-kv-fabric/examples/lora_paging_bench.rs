// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: MIT

//! Multi-tenant LoRA adapter paging demo at the fabric level
//! (HIP-1345 §2.4, Bounty #15 / engine#96, S-LoRA).
//!
//! Simulates 8 tenants x 2 adapters x 4 layers of 4 MiB pages (256 MiB of
//! demand) competing for a 40 MiB device budget — a standing eviction
//! storm. Phase A runs unpinned: every tenant thrashes. Phase B pins
//! tenant 0's adapters: its pages survive the storm while everyone else
//! keeps paying eviction cost. This is the policy-layer counterpart of
//! S-LoRA's tenant-isolation claim, measured honestly at the fabric level
//! (no device memory, no attention path — those belong to the engine).
//!
//! Run: `cargo run --release -p hanzo-kv-fabric --example lora_paging_bench`

use std::time::Instant;

use hanzo_kv_fabric::adapter_paging::{AdapterMeta, AdapterPagingError, AdapterPagingRegistry};
use hanzo_kv_fabric::compute_block_digest;

const MIB: u64 = 1024 * 1024;
const BUDGET_BYTES: u64 = 40 * MIB;
const PAGE_BYTES: u64 = 4 * MIB;
const TENANTS: u64 = 8;
const ADAPTERS_PER_TENANT: u64 = 2;
const LAYERS: u32 = 4;
const STORM_ROUNDS: usize = 32;
const PINNED_TENANT: u64 = 0;

fn tenant_salt(tenant: u64) -> u64 {
    1000 + tenant
}

fn adapter_name(tenant: u64, adapter: u64) -> String {
    format!("tenant-{tenant}/zen-adapter-{adapter}")
}

/// Deterministic stand-in layer payload digest, mirroring what a real
/// registration would pin for integrity verification.
fn fake_layer_digest(salt: u64, adapter: u64, layer: u32) -> [u8; 32] {
    let mut buf = Vec::with_capacity(8 + 8 + 4 + 64 * 1024);
    buf.extend_from_slice(&salt.to_le_bytes());
    buf.extend_from_slice(&adapter.to_le_bytes());
    buf.extend_from_slice(&layer.to_le_bytes());
    let mut state = salt ^ (adapter << 1) ^ u64::from(layer);
    for _ in 0..64 * 1024 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        buf.extend_from_slice(&(state as u32).to_le_bytes());
    }
    compute_block_digest(&buf)
}

fn register_all(registry: &AdapterPagingRegistry) {
    for tenant in 0..TENANTS {
        for adapter in 0..ADAPTERS_PER_TENANT {
            let salt = tenant_salt(tenant);
            let digests: Vec<[u8; 32]> = (0..LAYERS)
                .map(|layer| fake_layer_digest(salt, adapter, layer))
                .collect();
            registry
                .register_adapter(
                    AdapterMeta {
                        adapter_id: adapter_name(tenant, adapter),
                        cache_salt: salt,
                        num_layers: LAYERS,
                        bytes_per_layer: PAGE_BYTES,
                    },
                    digests,
                )
                .unwrap();
        }
    }
}

/// One full sweep of every tenant over every adapter layer, bucketing each
/// lookup into hit/miss (via stats deltas) and accumulating policy-layer
/// latency per bucket.
fn storm(
    registry: &AdapterPagingRegistry,
    rounds: usize,
) -> (u64, u64, u128, u128, Vec<u64>) {
    let mut hits_total = 0u64;
    let mut misses_total = 0u64;
    let mut hit_ns = 0u128;
    let mut miss_ns = 0u128;
    let mut hits_by_tenant = vec![0u64; TENANTS as usize];

    for _ in 0..rounds {
        for tenant in 0..TENANTS {
            for adapter in 0..ADAPTERS_PER_TENANT {
                for layer in 0..LAYERS {
                    let name = adapter_name(tenant, adapter);
                    let salt = tenant_salt(tenant);
                    let before = registry.stats();
                    let start = Instant::now();
                    let lease = registry.get_page(&name, salt, layer).unwrap();
                    let elapsed = start.elapsed().as_nanos();
                    let after = registry.stats();
                    if after.misses == before.misses {
                        hits_total += 1;
                        hit_ns += elapsed;
                        hits_by_tenant[tenant as usize] += 1;
                    } else {
                        misses_total += 1;
                        miss_ns += elapsed;
                    }
                    let _ = lease;
                }
            }
        }
    }
    (hits_total, misses_total, hit_ns, miss_ns, hits_by_tenant)
}

fn calls_per_tenant(rounds: usize) -> u64 {
    rounds as u64 * ADAPTERS_PER_TENANT * u64::from(LAYERS)
}

fn main() {
    let registry = AdapterPagingRegistry::new(BUDGET_BYTES);
    register_all(&registry);

    // Salt isolation sanity: the pinned tenant's name under a foreign salt
    // must not resolve to any registered adapter.
    let err = registry
        .get_page(&adapter_name(PINNED_TENANT, 0), tenant_salt(77), 0)
        .unwrap_err();
    assert!(matches!(err, AdapterPagingError::UnknownAdapter { .. }));

    println!(
        "LoRA adapter paging demo: {TENANTS} tenants x {ADAPTERS_PER_TENANT} adapters x {LAYERS} layers x {} MiB pages, {} MiB device budget",
        PAGE_BYTES / MIB,
        BUDGET_BYTES / MIB,
    );
    println!("=====================================================================");

    // Phase A: nobody pins; the storm thrashes everyone.
    let (hits_a, misses_a, hit_ns_a, miss_ns_a, _) = storm(&registry, STORM_ROUNDS);
    let total_a = hits_a + misses_a;
    println!(
        "phase A (unpinned)          : {hits_a} hits / {misses_a} misses ({:.1}% hit ratio), {} evictions",
        hits_a as f64 * 100.0 / total_a as f64,
        registry.stats().evictions,
    );

    // Phase B: pin the demo tenant's adapters, then re-run the storm.
    for adapter in 0..ADAPTERS_PER_TENANT {
        registry
            .pin_adapter(&adapter_name(PINNED_TENANT, adapter), tenant_salt(PINNED_TENANT))
            .unwrap();
    }
    let evictions_before_b = registry.stats().evictions;
    let (hits_b, misses_b, hit_ns_b, miss_ns_b, hits_by_tenant) = storm(&registry, STORM_ROUNDS);
    let evictions_b = registry.stats().evictions - evictions_before_b;

    let pinned_calls = calls_per_tenant(STORM_ROUNDS);
    let pinned_hits = hits_by_tenant[PINNED_TENANT as usize];
    let other_hits: u64 = hits_by_tenant
        .iter()
        .enumerate()
        .filter(|(t, _)| *t != PINNED_TENANT as usize)
        .map(|(_, h)| h)
        .sum();
    let other_calls = pinned_calls * (TENANTS - 1);

    println!(
        "phase B (tenant-{PINNED_TENANT} pinned)  : {hits_b} hits / {misses_b} misses, {evictions_b} evictions"
    );
    println!(
        "pinned tenant hit ratio     : {pinned_hits}/{pinned_calls} ({:.1}%)",
        pinned_hits as f64 * 100.0 / pinned_calls as f64
    );
    println!(
        "unpinned tenants hit ratio  : {other_hits}/{other_calls} ({:.1}%)",
        other_hits as f64 * 100.0 / other_calls as f64
    );
    println!(
        "policy cost, resident hit   : ~{} ns/lookup",
        if hits_a > 0 { hit_ns_a / hits_a as u128 } else { hit_ns_b / hits_b.max(1) as u128 }
    );
    println!(
        "policy cost, admission      : ~{} ns/lookup",
        if misses_a > 0 { miss_ns_a / misses_a as u128 } else { miss_ns_b / misses_b.max(1) as u128 }
    );

    // Pinned survival check: every pinned page is still resident (and
    // pinned) after the storm.
    let mut pinned_survivors = 0u32;
    for adapter in 0..ADAPTERS_PER_TENANT {
        for layer in 0..LAYERS {
            let lease = registry
                .get_page(&adapter_name(PINNED_TENANT, adapter), tenant_salt(PINNED_TENANT), layer)
                .unwrap();
            if lease.state == hanzo_kv_fabric::adapter_paging::PageState::Pinned {
                pinned_survivors += 1;
            }
        }
    }
    println!(
        "pinned pages after storm    : {pinned_survivors}/{} still Pinned",
        ADAPTERS_PER_TENANT * u64::from(LAYERS)
    );
    println!();
    println!("Unpinned, every tenant pays eviction cost on a shared budget;");
    println!("pinning one tenant's hot adapters makes its pages immortal under");
    println!("the storm while other tenants keep thrashing — tenant isolation");
    println!("at the paging-policy layer (HIP-1345 §2.4), no device I/O needed.");
}
