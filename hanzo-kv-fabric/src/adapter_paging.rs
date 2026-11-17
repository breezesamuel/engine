// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: MIT

//! Multi-tenant LoRA adapter paging for the KV fabric (HIP-1345 §2.3,
//! Bounty #15 / engine#96, S-LoRA <https://arxiv.org/abs/2311.03285>).
//!
//! Zen serves many fine-tuned LoRA adapters against shared base weights.
//! Adapter weights are per tenant, device memory is finite, so adapter
//! layer pages must move between host RAM and the device while requests
//! are in flight. This module is the fabric-level paging *policy* layer:
//!
//! - [`AdapterPagingRegistry`]: a device-byte-budgeted registry of adapter
//!   layer pages with LRU eviction and explicit pinning.
//! - [`salted_adapter_key`]: tenant-salted adapter identity, so two tenants
//!   registering adapters with identical names occupy disjoint page keys
//!   and can never share or observe each other's pages (HIP-1345 §1.1).
//! - [`PageLease`]: what the attention path receives on a hit; carries the
//!   registration-time SHA-256 digest so the consumer can verify the page
//!   payload before applying weights (fail closed on mismatch).
//!
//! Invariants enforced here:
//!
//! - **Salted keys (HIP-1345 §1.1)**: page identity is
//!   `(salted_adapter_key, layer)`. Identical adapter names under different
//!   tenant salts never collide, and a lease for one tenant's key can never
//!   be constructed from another tenant's salt.
//! - **Byte budget is a hard cap**: resident bytes never exceed the
//!   configured budget. Admission evicts the least-recently-used *unpinned*
//!   resident pages; when nothing is evictable, admission fails closed with
//!   [`AdapterPagingError::DeviceMemoryExhausted`] rather than
//!   over-subscribing device memory.
//! - **Pinning is authoritative**: pinned adapters' pages are never chosen
//!   as eviction victims, so a tenant can protect its hot adapter from
//!   thrashing under other tenants' load (DoS resistance).
//! - **Fail-closed config**: adapters with empty names, zero layers,
//!   zero-byte layers, or a digest vector whose length does not match
//!   `num_layers` are rejected at registration.
//!
//! What this module does NOT do: it does not touch device memory itself.
//! Storage, CUDA graph capture, and the attention-path RoPE translation for
//! position-shifted reuse live in the engine; this module brokers page
//! identity, residency, and budgeting so the scheduler can be integrated on
//! top without re-deriving the policy. Eviction victim selection is an
//! O(pages) scan, which is bounded at fabric scale (pages = registered
//! adapters x layers) and can be swapped for an intrusive LRU list when the
//! scheduler integration lands.

use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use xxhash_rust::xxh3::xxh3_64;

use thiserror::Error;

/// Compute a tenant-salted adapter key.
///
/// The tenant `cache_salt` is mixed into the buffer unconditionally
/// (HIP-1345 §1.1): identical adapter names under different tenants yield
/// different keys, so cross-tenant page sharing cannot occur by key
/// collision.
pub fn salted_adapter_key(adapter_id: &str, cache_salt: u64) -> u64 {
    let mut buf = Vec::with_capacity(adapter_id.len() + 8);
    buf.extend_from_slice(adapter_id.as_bytes());
    buf.extend_from_slice(&cache_salt.to_le_bytes());
    xxh3_64(&buf)
}

/// Residency state of one adapter layer page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageState {
    /// Page payload lives only in host RAM (not resident on the device).
    Hosted,
    /// Page is resident on the device and is an eviction candidate.
    Resident,
    /// Page is resident and pinned: never an eviction candidate.
    Pinned,
}

/// Static description of one registrable adapter.
///
/// `bytes_per_layer` is uniform across the adapter's layers in this bounded
/// phase; per-layer sizing arrives with the scheduler integration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterMeta {
    /// Tenant-scoped logical adapter name.
    pub adapter_id: String,
    /// Resolved tenant salt mixed into [`salted_adapter_key`].
    pub cache_salt: u64,
    /// Number of layers the adapter spans.
    pub num_layers: u32,
    /// Device bytes occupied by each layer page.
    pub bytes_per_layer: u64,
}

/// Lease handed to the attention path for one admitted or resident page.
///
/// `sha256` is the registration-time payload digest. Consumers MUST verify
/// the on-device payload against it before applying weights and MUST treat
/// a mismatch as a miss (recompute) rather than forcing reuse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageLease {
    pub adapter_key: u64,
    pub layer: u32,
    pub state: PageState,
    pub sha256: [u8; 32],
}

/// Cumulative policy-layer counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PagingStats {
    /// Lookups served by an already-resident page.
    pub hits: u64,
    /// Lookups that had to admit a hosted page.
    pub misses: u64,
    /// Resident pages demoted back to host RAM to make room.
    pub evictions: u64,
    /// Device bytes currently charged to resident (or pinned) pages.
    pub resident_bytes: u64,
}

/// Errors surfaced by [`AdapterPagingRegistry`]. All failure modes fail
/// closed: no partial admission, no implicit sharing, no over-subscription.
#[derive(Debug, Error)]
pub enum AdapterPagingError {
    #[error("adapter {name:?} is not registered under salt {salt}")]
    UnknownAdapter { name: String, salt: u64 },
    #[error(
        "adapter {name:?} under salt {salt} has {num_layers} layers; layer {layer} is out of range"
    )]
    LayerOutOfRange {
        name: String,
        salt: u64,
        num_layers: u32,
        layer: u32,
    },
    #[error("page needs {need} bytes but the device budget is {budget} bytes")]
    OutOfBudget { need: u64, budget: u64 },
    #[error(
        "device memory exhausted: {resident} resident bytes, {need} needed, no evictable unpinned pages"
    )]
    DeviceMemoryExhausted { resident: u64, need: u64 },
    #[error("adapter {name:?} under salt {salt} is already registered")]
    AdapterAlreadyRegistered { name: String, salt: u64 },
    #[error("adapter config invalid: {reason}")]
    InvalidAdapterConfig { reason: String },
}

#[derive(Debug)]
struct RegisteredAdapter {
    meta: AdapterMeta,
    digests: Vec<[u8; 32]>,
    pinned: bool,
}

#[derive(Debug)]
struct PageEntry {
    bytes: u64,
    state: PageState,
    last_use: u64,
    sha256: [u8; 32],
}

#[derive(Debug, Default)]
struct Inner {
    adapters: FxHashMap<u64, RegisteredAdapter>,
    pages: FxHashMap<(u64, u32), PageEntry>,
    tick: u64,
    stats: PagingStats,
}

/// Device-byte-budgeted multi-tenant adapter page registry.
///
/// The registry is shared across tenants (that is the point: one device
/// budget serves everyone), but page identity is salted per tenant, so
/// sharing the budget never means sharing content. All methods take
/// `&self` and are internally locked, mirroring [`crate::SegmentDirectory`].
#[derive(Debug)]
pub struct AdapterPagingRegistry {
    budget_bytes: u64,
    inner: RwLock<Inner>,
}

impl AdapterPagingRegistry {
    /// Create a registry whose resident pages never exceed `budget_bytes`.
    pub fn new(budget_bytes: u64) -> Self {
        Self {
            budget_bytes,
            inner: RwLock::new(Inner::default()),
        }
    }

    /// The hard device-byte cap this registry enforces.
    pub fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    /// Register an adapter and its per-layer payload digests.
    ///
    /// All pages start [`PageState::Hosted`]; nothing is resident until a
    /// [`Self::get_page`] admission. Returns the salted adapter key.
    pub fn register_adapter(
        &self,
        meta: AdapterMeta,
        layer_digests: Vec<[u8; 32]>,
    ) -> Result<u64, AdapterPagingError> {
        if meta.adapter_id.is_empty() {
            return Err(AdapterPagingError::InvalidAdapterConfig {
                reason: "adapter_id must not be empty".to_string(),
            });
        }
        if meta.num_layers == 0 {
            return Err(AdapterPagingError::InvalidAdapterConfig {
                reason: format!("adapter {:?} must span at least one layer", meta.adapter_id),
            });
        }
        if meta.bytes_per_layer == 0 {
            return Err(AdapterPagingError::InvalidAdapterConfig {
                reason: format!("adapter {:?} layers must occupy bytes", meta.adapter_id),
            });
        }
        if layer_digests.len() != meta.num_layers as usize {
            return Err(AdapterPagingError::InvalidAdapterConfig {
                reason: format!(
                    "adapter {:?} declares {} layers but {} digests were supplied",
                    meta.adapter_id,
                    meta.num_layers,
                    layer_digests.len()
                ),
            });
        }
        let key = salted_adapter_key(&meta.adapter_id, meta.cache_salt);
        let mut inner = self.inner.write();
        if inner.adapters.contains_key(&key) {
            return Err(AdapterPagingError::AdapterAlreadyRegistered {
                name: meta.adapter_id.clone(),
                salt: meta.cache_salt,
            });
        }
        let bytes_per_layer = meta.bytes_per_layer;
        for (layer, sha256) in layer_digests.iter().enumerate() {
            inner.pages.insert(
                (key, layer as u32),
                PageEntry {
                    bytes: bytes_per_layer,
                    state: PageState::Hosted,
                    last_use: 0,
                    sha256: *sha256,
                },
            );
        }
        inner.adapters.insert(
            key,
            RegisteredAdapter {
                meta,
                digests: layer_digests,
                pinned: false,
            },
        );
        Ok(key)
    }

    /// Remove an adapter and all of its pages, releasing any resident bytes.
    pub fn deregister_adapter(
        &self,
        adapter_id: &str,
        cache_salt: u64,
    ) -> Result<(), AdapterPagingError> {
        let key = salted_adapter_key(adapter_id, cache_salt);
        let mut inner = self.inner.write();
        let adapter = inner
            .adapters
            .remove(&key)
            .ok_or_else(|| AdapterPagingError::UnknownAdapter {
                name: adapter_id.to_string(),
                salt: cache_salt,
            })?;
        for layer in 0..adapter.meta.num_layers {
            if let Some(entry) = inner.pages.remove(&(key, layer)) {
                if entry.state != PageState::Hosted {
                    inner.stats.resident_bytes -= entry.bytes;
                }
            }
        }
        Ok(())
    }

    /// Pin an adapter: its pages are never eviction candidates until
    /// [`Self::unpin_adapter`].
    pub fn pin_adapter(&self, adapter_id: &str, cache_salt: u64) -> Result<(), AdapterPagingError> {
        let key = salted_adapter_key(adapter_id, cache_salt);
        let mut inner = self.inner.write();
        let num_layers = {
            let adapter = inner
                .adapters
                .get_mut(&key)
                .ok_or_else(|| AdapterPagingError::UnknownAdapter {
                    name: adapter_id.to_string(),
                    salt: cache_salt,
                })?;
            adapter.pinned = true;
            adapter.meta.num_layers
        };
        for layer in 0..num_layers {
            if let Some(entry) = inner.pages.get_mut(&(key, layer)) {
                if entry.state == PageState::Resident {
                    entry.state = PageState::Pinned;
                }
            }
        }
        Ok(())
    }

    /// Unpin an adapter, restoring its resident pages to evictability.
    pub fn unpin_adapter(
        &self,
        adapter_id: &str,
        cache_salt: u64,
    ) -> Result<(), AdapterPagingError> {
        let key = salted_adapter_key(adapter_id, cache_salt);
        let mut inner = self.inner.write();
        let num_layers = {
            let adapter = inner
                .adapters
                .get_mut(&key)
                .ok_or_else(|| AdapterPagingError::UnknownAdapter {
                    name: adapter_id.to_string(),
                    salt: cache_salt,
                })?;
            adapter.pinned = false;
            adapter.meta.num_layers
        };
        for layer in 0..num_layers {
            if let Some(entry) = inner.pages.get_mut(&(key, layer)) {
                if entry.state == PageState::Pinned {
                    entry.state = PageState::Resident;
                }
            }
        }
        Ok(())
    }

    /// Look up one adapter layer page, admitting it from host RAM if
    /// hosted.
    ///
    /// Admission evicts least-recently-used unpinned resident pages until
    /// the hard budget accommodates the incoming page, then fails closed
    /// with [`AdapterPagingError::DeviceMemoryExhausted`] if nothing is
    /// evictable. Hits refresh the page's LRU position; the returned lease
    /// carries the registration-time digest for consumer-side integrity
    /// verification.
    pub fn get_page(
        &self,
        adapter_id: &str,
        cache_salt: u64,
        layer: u32,
    ) -> Result<PageLease, AdapterPagingError> {
        let key = salted_adapter_key(adapter_id, cache_salt);
        let mut inner = self.inner.write();

        let pinned = {
            let adapter = inner
                .adapters
                .get(&key)
                .ok_or_else(|| AdapterPagingError::UnknownAdapter {
                    name: adapter_id.to_string(),
                    salt: cache_salt,
                })?;
            if layer >= adapter.meta.num_layers {
                return Err(AdapterPagingError::LayerOutOfRange {
                    name: adapter_id.to_string(),
                    salt: cache_salt,
                    num_layers: adapter.meta.num_layers,
                    layer,
                });
            }
            adapter.pinned
        };

        inner.tick += 1;
        let tick = inner.tick;

        // Destructure the guard so `pages` and `stats` are disjoint places;
        // the eviction path interleaves access to both.
        let Inner { pages, stats, .. } = &mut *inner;

        let hit = {
            let entry = pages
                .get_mut(&(key, layer))
                .expect("registered adapter must own every layer page");
            let sha256 = entry.sha256;
            if entry.state != PageState::Hosted {
                entry.last_use = tick;
                stats.hits += 1;
                Some((entry.state, sha256))
            } else {
                None
            }
        };
        if let Some((state, sha256)) = hit {
            return Ok(PageLease {
                adapter_key: key,
                layer,
                state,
                sha256,
            });
        }

        let need = {
            let entry = pages
                .get(&(key, layer))
                .expect("registered adapter must own every layer page");
            entry.bytes
        };
        if need > self.budget_bytes {
            return Err(AdapterPagingError::OutOfBudget {
                need,
                budget: self.budget_bytes,
            });
        }

        while stats.resident_bytes + need > self.budget_bytes {
            let victim = pages
                .iter()
                .filter(|(_, e)| e.state == PageState::Resident)
                .min_by_key(|(_, e)| e.last_use)
                .map(|(k, _)| *k);
            let Some(victim_key) = victim else {
                return Err(AdapterPagingError::DeviceMemoryExhausted {
                    resident: stats.resident_bytes,
                    need,
                });
            };
            let victim_entry = pages
                .get_mut(&victim_key)
                .expect("victim key came from the live page map");
            victim_entry.state = PageState::Hosted;
            stats.resident_bytes -= victim_entry.bytes;
            stats.evictions += 1;
        }

        let entry = pages
            .get_mut(&(key, layer))
            .expect("registered adapter must own every layer page");
        entry.state = if pinned {
            PageState::Pinned
        } else {
            PageState::Resident
        };
        entry.last_use = tick;
        stats.resident_bytes += need;
        stats.misses += 1;
        Ok(PageLease {
            adapter_key: key,
            layer,
            state: entry.state,
            sha256: entry.sha256,
        })
    }

    /// Snapshot of the cumulative policy counters.
    pub fn stats(&self) -> PagingStats {
        self.inner.read().stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE_BYTES: u64 = 1024 * 1024;

    fn digest(tag: u8) -> [u8; 32] {
        let mut d = [0u8; 32];
        d[0] = tag;
        d
    }

    fn meta(name: &str, salt: u64, layers: u32) -> AdapterMeta {
        AdapterMeta {
            adapter_id: name.to_string(),
            cache_salt: salt,
            num_layers: layers,
            bytes_per_layer: PAGE_BYTES,
        }
    }

    #[test]
    fn salted_adapter_key_separates_tenants_and_names() {
        let a = salted_adapter_key("shared-lora", 1);
        let b = salted_adapter_key("shared-lora", 2);
        let c = salted_adapter_key("other-lora", 1);
        assert_ne!(a, b, "salt must separate tenants (HIP-1345 §1.1)");
        assert_ne!(a, c, "adapter name must partake in the key");
        assert_eq!(a, salted_adapter_key("shared-lora", 1), "keys are stable");
    }

    #[test]
    fn register_rejects_invalid_configs_fail_closed() {
        let reg = AdapterPagingRegistry::new(8 * PAGE_BYTES);

        let err = reg
            .register_adapter(meta("", 1, 2), vec![digest(0), digest(1)])
            .unwrap_err();
        assert!(matches!(err, AdapterPagingError::InvalidAdapterConfig { .. }));

        let err = reg
            .register_adapter(meta("zero-layers", 1, 0), vec![])
            .unwrap_err();
        assert!(matches!(err, AdapterPagingError::InvalidAdapterConfig { .. }));

        let err = reg
            .register_adapter(
                AdapterMeta {
                    bytes_per_layer: 0,
                    ..meta("zero-bytes", 1, 1)
                },
                vec![digest(0)],
            )
            .unwrap_err();
        assert!(matches!(err, AdapterPagingError::InvalidAdapterConfig { .. }));

        let err = reg
            .register_adapter(meta("digest-mismatch", 1, 2), vec![digest(0)])
            .unwrap_err();
        assert!(matches!(err, AdapterPagingError::InvalidAdapterConfig { .. }));
    }

    #[test]
    fn duplicate_registration_is_rejected() {
        let reg = AdapterPagingRegistry::new(8 * PAGE_BYTES);
        reg.register_adapter(meta("dup", 7, 1), vec![digest(0)]).unwrap();
        let err = reg
            .register_adapter(meta("dup", 7, 1), vec![digest(0)])
            .unwrap_err();
        assert!(matches!(err, AdapterPagingError::AdapterAlreadyRegistered { .. }));
        // Same name, different tenant salt: disjoint keys, both fine.
        reg.register_adapter(meta("dup", 8, 1), vec![digest(0)]).unwrap();
    }

    #[test]
    fn get_page_admits_then_hits_and_touches_lru() {
        let reg = AdapterPagingRegistry::new(4 * PAGE_BYTES);
        reg.register_adapter(meta("a", 1, 4), (0..4).map(digest).collect())
            .unwrap();

        let lease = reg.get_page("a", 1, 0).unwrap();
        assert_eq!(lease.state, PageState::Resident);
        assert_eq!(lease.sha256, digest(0));
        let stats = reg.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.resident_bytes, PAGE_BYTES);

        let lease = reg.get_page("a", 1, 0).unwrap();
        assert_eq!(lease.state, PageState::Resident);
        let stats = reg.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.resident_bytes, PAGE_BYTES);
    }

    #[test]
    fn out_of_range_layer_fails_closed() {
        let reg = AdapterPagingRegistry::new(4 * PAGE_BYTES);
        reg.register_adapter(meta("a", 1, 2), vec![digest(0), digest(1)])
            .unwrap();
        let err = reg.get_page("a", 1, 2).unwrap_err();
        assert!(matches!(err, AdapterPagingError::LayerOutOfRange { .. }));
        let err = reg.get_page("missing", 1, 0).unwrap_err();
        assert!(matches!(err, AdapterPagingError::UnknownAdapter { .. }));
    }

    #[test]
    fn lru_eviction_respects_the_hard_budget() {
        let reg = AdapterPagingRegistry::new(2 * PAGE_BYTES);
        reg.register_adapter(meta("a", 1, 4), (0..4).map(digest).collect())
            .unwrap();

        reg.get_page("a", 1, 0).unwrap();
        reg.get_page("a", 1, 1).unwrap();
        assert_eq!(reg.stats().resident_bytes, 2 * PAGE_BYTES);

        // Third admission must evict the LRU page (layer 0).
        reg.get_page("a", 1, 2).unwrap();
        let stats = reg.stats();
        assert_eq!(stats.evictions, 1);
        assert_eq!(stats.resident_bytes, 2 * PAGE_BYTES, "budget is a hard cap");

        // Layer 0 was evicted: re-admission is a miss again.
        reg.get_page("a", 1, 0).unwrap();
        let stats = reg.stats();
        assert_eq!(stats.evictions, 2);
        assert_eq!(stats.hits, 0, "evicted page must not report a hit");
        assert_eq!(stats.resident_bytes, 2 * PAGE_BYTES);
    }

    #[test]
    fn single_page_larger_than_budget_fails_closed() {
        let reg = AdapterPagingRegistry::new(PAGE_BYTES);
        reg.register_adapter(
            AdapterMeta {
                bytes_per_layer: 2 * PAGE_BYTES,
                ..meta("huge", 1, 1)
            },
            vec![digest(0)],
        )
        .unwrap();
        let err = reg.get_page("huge", 1, 0).unwrap_err();
        assert!(matches!(err, AdapterPagingError::OutOfBudget { .. }));
        assert_eq!(reg.stats().resident_bytes, 0);
    }

    #[test]
    fn pinned_pages_are_never_evicted_and_protect_their_tenant() {
        let reg = AdapterPagingRegistry::new(3 * PAGE_BYTES);
        reg.register_adapter(meta("hot", 1, 3), (0..3).map(digest).collect())
            .unwrap();
        reg.register_adapter(meta("cold-a", 2, 3), (0..3).map(digest).collect())
            .unwrap();
        reg.register_adapter(meta("cold-b", 3, 3), (0..3).map(digest).collect())
            .unwrap();

        reg.pin_adapter("hot", 1).unwrap();
        for layer in 0..3 {
            let lease = reg.get_page("hot", 1, layer).unwrap();
            assert_eq!(lease.state, PageState::Pinned, "pinned admissions pin");
        }
        assert_eq!(reg.stats().resident_bytes, 3 * PAGE_BYTES);

        // Every other tenant's admission now has no evictable victim.
        let err = reg.get_page("cold-a", 2, 0).unwrap_err();
        assert!(
            matches!(err, AdapterPagingError::DeviceMemoryExhausted { .. }),
            "pinned tenant must not be silently evicted (DoS resistance)"
        );
        assert_eq!(reg.stats().resident_bytes, 3 * PAGE_BYTES);

        // Unpinning restores evictability.
        reg.unpin_adapter("hot", 1).unwrap();
        reg.get_page("cold-a", 2, 0).unwrap();
        assert_eq!(reg.stats().evictions, 1);
    }

    #[test]
    fn tenant_salts_isolate_pages_under_a_shared_budget() {
        let reg = AdapterPagingRegistry::new(2 * PAGE_BYTES);
        reg.register_adapter(meta("same-name", 1, 1), vec![digest(1)])
            .unwrap();
        reg.register_adapter(meta("same-name", 2, 1), vec![digest(2)])
            .unwrap();

        let lease_a = reg.get_page("same-name", 1, 0).unwrap();
        let lease_b = reg.get_page("same-name", 2, 0).unwrap();
        assert_ne!(
            lease_a.adapter_key, lease_b.adapter_key,
            "identical names under different salts occupy disjoint pages"
        );
        assert_eq!(lease_a.sha256, digest(1));
        assert_eq!(lease_b.sha256, digest(2));

        // Deregistering one tenant never touches the other.
        reg.deregister_adapter("same-name", 1).unwrap();
        assert!(reg.get_page("same-name", 1, 0).is_err());
        let lease = reg.get_page("same-name", 2, 0).unwrap();
        assert_eq!(lease.state, PageState::Resident);
        assert_eq!(reg.stats().hits, 1, "tenant B's page was never evicted");

        // Deregistering released tenant A's resident bytes into the budget.
        reg.deregister_adapter("same-name", 2).unwrap();
        assert_eq!(reg.stats().resident_bytes, 0);
    }

    #[test]
    fn deregister_frees_budget_for_new_admissions() {
        let reg = AdapterPagingRegistry::new(2 * PAGE_BYTES);
        reg.register_adapter(meta("big", 1, 2), vec![digest(0), digest(1)])
            .unwrap();
        reg.get_page("big", 1, 0).unwrap();
        reg.get_page("big", 1, 1).unwrap();
        assert_eq!(reg.stats().resident_bytes, 2 * PAGE_BYTES);

        reg.deregister_adapter("big", 1).unwrap();
        assert_eq!(reg.stats().resident_bytes, 0);

        reg.register_adapter(meta("next", 9, 2), vec![digest(2), digest(3)])
            .unwrap();
        reg.get_page("next", 9, 0).unwrap();
        reg.get_page("next", 9, 1).unwrap();
        assert_eq!(reg.stats().resident_bytes, 2 * PAGE_BYTES);
    }

    #[test]
    fn unpin_restores_evictability() {
        let reg = AdapterPagingRegistry::new(2 * PAGE_BYTES);
        reg.register_adapter(meta("p", 1, 2), vec![digest(0), digest(1)])
            .unwrap();
        reg.register_adapter(meta("q", 2, 2), vec![digest(2), digest(3)])
            .unwrap();

        reg.pin_adapter("p", 1).unwrap();
        reg.get_page("p", 1, 0).unwrap();
        reg.get_page("p", 1, 1).unwrap();
        let err = reg.get_page("q", 2, 0).unwrap_err();
        assert!(matches!(err, AdapterPagingError::DeviceMemoryExhausted { .. }));

        reg.unpin_adapter("p", 1).unwrap();
        reg.get_page("q", 2, 0).unwrap();
        assert_eq!(reg.stats().evictions, 1, "unpinned pages became victims");
    }

    #[test]
    fn concurrent_access_never_exceeds_the_budget() {
        let reg = std::sync::Arc::new(AdapterPagingRegistry::new(4 * PAGE_BYTES));
        let tenants: Vec<(String, u64)> = (0..8)
            .map(|t| (format!("tenant-{t}/lora"), 100 + t))
            .collect();
        for (name, salt) in &tenants {
            reg.register_adapter(
                meta(name, *salt, 4),
                (0..4).map(digest).collect(),
            )
            .unwrap();
        }

        std::thread::scope(|scope| {
            for (name, salt) in &tenants {
                let reg = std::sync::Arc::clone(&reg);
                let name = name.clone();
                let salt = *salt;
                scope.spawn(move || {
                    for round in 0..100u32 {
                        let layer = (round % 4) as u32;
                        // The shared budget forces constant eviction pressure;
                        // every call must either succeed or fail closed with
                        // a well-formed error, never over-subscribe.
                        let _ = reg.get_page(&name, salt, layer);
                    }
                });
            }
        });

        let stats = reg.stats();
        assert!(
            stats.resident_bytes <= reg.budget_bytes(),
            "resident bytes {} exceeded budget {}",
            stats.resident_bytes,
            reg.budget_bytes()
        );
        assert!(stats.hits + stats.misses > 0);
    }
}
