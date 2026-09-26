//! Persistent per-server performance calibration (phase 4B.3).
//!
//! The 4B.2 study measured the cost model it drove and found the weak link:
//! predictions ran on startup seeds for the requests that mattered most —
//! every burst arrival — because the online model only learns *from*
//! completed requests, and each benchmark leg (and each gateway restart)
//! began from seeds again. Measured cost: TTFT prediction MAE 170–360 s with
//! bias ≈ +MAE (reality was always worse than priced), because the seeded
//! decode rate (~25 ms/tok) under-prices loaded decode by 10–13×.
//!
//! The fix has two halves:
//!
//! 1. **Persist** what the online model learns, keyed by the deployment
//!    tuple that determines the physics — machine × model × runtime × slot
//!    count — so the next process starts from measurement, not seeds:
//!
//!    ```text
//!    calibration/
//!      apple-m5/
//!        qwen3.5-9b-q4_k_m.gguf/
//!          llama-cpp-8slots.json
//!    ```
//!
//! 2. **Stop pricing all load levels with one decode rate.** One server-side
//!    number cannot describe a machine where solo decode runs ~25 ms/tok and
//!    8-way concurrent decode runs ~10× that. Rates are maintained per
//!    *concurrency bucket* of active server-side streams
//!    (`["1", "2", "3-4", "5-8"]`) with an EWMA each, and prefill carries a
//!    measured contention multiplier per bucket (Metal prefill anti-scales
//!    under parallel prefills — phase 4B.2). No invented constants: every
//!    number in the file is an EWMA of this machine's own observations.
//!
//! One [`RuntimeCalibration`] is shared by every [`crate::WorkerHandle`]
//! that models the same server process: its prefill regression, decode
//! buckets and active-dispatch registry are *server-wide* state (eight slot
//! handles share one GPU, one memory bus, one Metal queue).

use crate::{clamp, ServiceDefaults};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Bump when the profile layout changes incompatibly.
pub const SCHEMA_VERSION: u32 = 1;

/// Concurrency buckets for server-side stream counts, in upper-bound order.
/// Bucket `i` covers `lower[i]..=lower[i+1]-1` (last bucket saturates).
pub const BUCKET_LABELS: [&str; 4] = ["1", "2", "3-4", "5-8"];
const BUCKET_LOWER: [usize; 4] = [1, 2, 3, 5];

/// Index of the bucket an active-stream count falls into.
pub fn bucket_index(active: usize) -> usize {
    let mut i = 0;
    while i + 1 < BUCKET_LOWER.len() && active >= BUCKET_LOWER[i + 1] {
        i += 1;
    }
    i
}

/// Human label ("1" | "2" | "3-4" | "5-8") of an active-stream count.
pub fn bucket_label(active: usize) -> &'static str {
    BUCKET_LABELS[bucket_index(active)]
}

/// Filesystem-safe key fragment for a model identifier.
pub fn slugify(model: &str) -> String {
    let mut out = String::new();
    let mut last_dash = false;
    for ch in model.trim().to_lowercase().chars() {
        let c = match ch {
            'a'..='z' | '0'..='9' | '.' | '_' => ch,
            _ => '-',
        };
        if c == '-' && last_dash {
            continue;
        }
        last_dash = c == '-';
        out.push(c);
    }
    let trimmed = out.trim_matches('-').to_string();
    let mut s: String = trimmed.chars().take(64).collect();
    if s.is_empty() {
        s = "model".into();
    }
    s
}

fn epoch_s() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Profile (the persisted form)
// ---------------------------------------------------------------------------

/// Prefill side: solo-rate regression plus measured contention multipliers.
/// `effective_prefill_ms ≈ fixed + tokens × ms_per_token × multiplier(bucket)`
/// where the bucket counts concurrent prefills *including* this one.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PrefillCalibration {
    pub fixed_ms: f64,
    pub ms_per_token: f64,
    pub samples: u64,
    /// Per-bucket slowdown multipliers over the solo rate, positional over
    /// [`BUCKET_LABELS`]. 1.0 = no measured penalty.
    pub contention_multiplier: [f64; 4],
    pub contention_samples: [u64; 4],
}

/// Decode side: ms/token per active-decode-stream bucket (positional over
/// [`BUCKET_LABELS`]). The 4B.2 lesson: this is where one-scalar models die —
/// loaded decode cost 10–13× solo on this machine.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DecodeCalibration {
    pub ms_per_token: [f64; 4],
    pub samples: [u64; 4],
}

/// The persisted profile. Everything in it is measured (EWMA state), plus
/// the deployment tuple it belongs to.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CalibrationProfile {
    pub schema_version: u32,
    /// Self-describing bucket order: ["1", "2", "3-4", "5-8"].
    pub bucket_labels: [String; 4],
    pub machine: String,
    pub runtime: String,
    pub model: String,
    pub slot_count: u32,
    pub updated_at_epoch_s: u64,
    pub prefill: PrefillCalibration,
    pub decode: DecodeCalibration,
}

// ---------------------------------------------------------------------------
// Store (where the profile lives)
// ---------------------------------------------------------------------------

/// The deployment tuple that keys one calibration file.
#[derive(Debug, Clone)]
pub struct CalibrationKey {
    pub machine: String,
    pub runtime: String,
    pub model_slug: String,
    pub slot_count: u32,
}

impl CalibrationKey {
    pub fn new(machine: &str, runtime: &str, model: &str, slot_count: u32) -> Self {
        Self {
            machine: machine.to_string(),
            runtime: runtime.to_string(),
            model_slug: slugify(model),
            slot_count,
        }
    }

    fn file_name(&self) -> String {
        if self.slot_count > 0 {
            format!("{}-{}slots.json", self.runtime, self.slot_count)
        } else {
            format!("{}-shared.json", self.runtime)
        }
    }

    /// `<dir>/<machine>/<model_slug>/<runtime>-<n>slots.json`
    pub fn path_under(&self, dir: &Path) -> PathBuf {
        dir.join(&self.machine).join(&self.model_slug).join(self.file_name())
    }
}

/// Loads and atomically stores one calibration JSON file.
pub struct CalibrationStore {
    path: PathBuf,
}

impl CalibrationStore {
    pub fn new(dir: &Path, key: &CalibrationKey) -> Self {
        Self { path: key.path_under(dir) }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// `None` when no profile exists yet (fresh deployment). A corrupt file
    /// is reported and treated as absent — a bad calibration must never
    /// wedge the gateway.
    pub fn load(&self) -> Option<CalibrationProfile> {
        let text = std::fs::read_to_string(&self.path).ok()?;
        match serde_json::from_str(&text) {
            Ok(p) => Some(p),
            Err(e) => {
                tracing::warn!(path = %self.path.display(), error = %e,
                    "calibration file corrupt — ignoring (will re-seed and overwrite)");
                None
            }
        }
    }

    /// Write-then-rename so a crash mid-save never leaves a truncated file.
    pub fn save(&self, profile: &CalibrationProfile) -> std::io::Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(profile).unwrap_or_default())?;
        std::fs::rename(&tmp, &self.path)
    }
}

// ---------------------------------------------------------------------------
// Online models (server-wide state, shared by all handles of one server)
// ---------------------------------------------------------------------------

/// EWMA update weight for the prefill regression: small on purpose — it
/// solves a 2×2 system per update and needs a long, stable memory to
/// separate fixed overhead from per-token cost.
const PREFILL_ALPHA: f64 = 0.05;
/// Decode buckets see few samples per burst (≈8 per c8 leg); they must
/// converge within one leg to be useful in the next.
const DECODE_ALPHA: f64 = 0.25;
/// Contention multipliers: same reasoning as decode buckets.
const CONTENTION_ALPHA: f64 = 0.2;

fn ewma_step(current: f64, obs: f64, alpha: f64) -> f64 {
    current + alpha * (obs - current)
}

/// Exponentially-weighted online least squares for `y = fixed + rate·t`
/// (prefill time vs input tokens). Solving the normal equations each update
/// is exact and unconditionally stable, unlike per-parameter gradient steps
/// which oscillate on this problem.
struct OnlineRateModel {
    fixed: f64,
    rate: f64,
    // EWMA moments: E[1], E[t], E[t²], E[y], E[t·y]
    m_n: f64,
    m_t: f64,
    m_tt: f64,
    m_y: f64,
    m_ty: f64,
    samples: u64,
    /// Minimum weighted variance of t before the split is identifiable.
    min_t_variance: f64,
}

impl OnlineRateModel {
    fn seeded(fixed: f64, rate: f64) -> Self {
        Self {
            fixed,
            rate,
            m_n: 1.0,
            m_t: 0.0,
            m_tt: 0.0,
            m_y: 0.0,
            m_ty: 0.0,
            samples: 0,
            min_t_variance: 1.0,
        }
    }

    /// Feed one (t, y) observation; refits fixed/rate when the observed
    /// prompt sizes have enough spread to identify both parameters.
    fn observe(&mut self, t: f64, y: f64) {
        self.samples += 1;
        self.m_n = ewma_step(self.m_n, 1.0, PREFILL_ALPHA);
        self.m_t = ewma_step(self.m_t, t, PREFILL_ALPHA);
        self.m_tt = ewma_step(self.m_tt, t * t, PREFILL_ALPHA);
        self.m_y = ewma_step(self.m_y, y, PREFILL_ALPHA);
        self.m_ty = ewma_step(self.m_ty, t * y, PREFILL_ALPHA);

        let (n, mt, mtt, my, mty) = (self.m_n, self.m_t, self.m_tt, self.m_y, self.m_ty);
        let var_t = mtt - mt * mt / n;
        if var_t < self.min_t_variance * n {
            return; // not enough spread in prompt sizes yet — keep seeds
        }
        self.rate = clamp((mty - mt * my / n) / var_t, 0.1, 50.0);
        self.fixed = clamp((my - self.rate * mt) / n, 0.0, 10_000.0);
    }
}

// ---------------------------------------------------------------------------
// Dispatch registry (server-wide active-stream counting)
// ---------------------------------------------------------------------------

/// One in-flight dispatch, minted at reservation time. Replaces the raw
/// `(Instant, est_ms)` pair the router used to pass around: the `id` makes
/// completion matching exact (two `Instant::now()` calls can land on the
/// same clock tick) and carries the estimate the pricing used.
#[derive(Debug, Clone, Copy)]
pub struct DispatchTicket {
    pub start: Instant,
    pub id: u64,
    pub est_total_ms: f64,
    /// The prefill portion of `est_total_ms`: a dispatch younger than this
    /// is (still estimated to be) prefilling — the phase classifier for
    /// active-stream counting.
    pub est_prefill_ms: f64,
}

struct RegistryEntry {
    id: u64,
    started: Instant,
    est_prefill_ms: f64,
}

struct CalibInner {
    prefill: OnlineRateModel,
    decode: [f64; 4],
    decode_samples: [u64; 4],
    contention: [f64; 4],
    contention_samples: [u64; 4],
    seed_decode_ms_per_tok: f64,
    registry: Vec<RegistryEntry>,
    next_id: u64,
}

/// Measured-bucket lookup with the fallback chain: exact bucket → nearest
/// higher (over-pricing loaded queues is the safe direction — 4B.2's
/// failure was under-pricing them) → nearest lower → `fallback`.
fn bucket_lookup(values: &[f64; 4], samples: &[u64; 4], want: usize, fallback: f64) -> f64 {
    if samples[want] > 0 {
        return values[want];
    }
    for j in (want + 1)..4 {
        if samples[j] > 0 {
            return values[j];
        }
    }
    for j in (0..want).rev() {
        if samples[j] > 0 {
            return values[j];
        }
    }
    fallback
}

fn decode_locked(g: &CalibInner, active_streams: usize) -> f64 {
    bucket_lookup(
        &g.decode,
        &g.decode_samples,
        bucket_index(active_streams.max(1)),
        g.seed_decode_ms_per_tok,
    )
}

fn contention_locked(g: &CalibInner, active_prefills: usize) -> f64 {
    bucket_lookup(
        &g.contention,
        &g.contention_samples,
        bucket_index(active_prefills.max(1)),
        1.0,
    )
}

static SERVER_SEQ: AtomicU64 = AtomicU64::new(1);

/// Shared, persistent performance model of ONE server process (all its
/// slots). See the module docs.
pub struct RuntimeCalibration {
    /// Stable identity for logs/status ("server 1").
    pub server_id: u64,
    machine: String,
    runtime: String,
    model: String,
    slot_count: u32,
    inner: Mutex<CalibInner>,
    dirty: AtomicBool,
    loaded_from_store: bool,
}

impl RuntimeCalibration {
    /// A fresh model from phase-2-style seed values (no persisted state).
    pub fn from_seeds(
        machine: &str,
        runtime: &str,
        model: &str,
        slot_count: u32,
        defaults: ServiceDefaults,
    ) -> Self {
        let inner = CalibInner {
            prefill: OnlineRateModel::seeded(defaults.fixed_ms, defaults.prefill_ms_per_tok),
            decode: [defaults.decode_ms_per_tok; 4],
            decode_samples: [0; 4],
            contention: [1.0; 4],
            contention_samples: [0; 4],
            seed_decode_ms_per_tok: defaults.decode_ms_per_tok,
            registry: Vec::new(),
            next_id: 1,
        };
        Self {
            server_id: SERVER_SEQ.fetch_add(1, Ordering::Relaxed),
            machine: machine.to_string(),
            runtime: runtime.to_string(),
            model: model.to_string(),
            slot_count,
            inner: Mutex::new(inner),
            dirty: AtomicBool::new(false),
            loaded_from_store: false,
        }
    }

    /// Adopt a persisted profile as current state. Profile values become
    /// both the working rates AND their evidence (their samples ride along),
    /// so loaded buckets are immediately trusted; physical fallback seeds
    /// still come from `defaults` for anything the profile left empty. The
    /// regression's EWMA moments restart empty — its (fixed, rate) stay at
    /// the adopted values until fresh observations re-identify them.
    pub fn from_profile(profile: &CalibrationProfile, defaults: ServiceDefaults) -> Self {
        let inner = CalibInner {
            prefill: OnlineRateModel {
                fixed: profile.prefill.fixed_ms,
                rate: profile.prefill.ms_per_token,
                m_n: 1.0,
                m_t: 0.0,
                m_tt: 0.0,
                m_y: 0.0,
                m_ty: 0.0,
                samples: profile.prefill.samples,
                min_t_variance: 1.0,
            },
            decode: profile.decode.ms_per_token,
            decode_samples: profile.decode.samples,
            contention: profile.prefill.contention_multiplier,
            contention_samples: profile.prefill.contention_samples,
            seed_decode_ms_per_tok: defaults.decode_ms_per_tok,
            registry: Vec::new(),
            next_id: 1,
        };
        Self {
            server_id: SERVER_SEQ.fetch_add(1, Ordering::Relaxed),
            machine: profile.machine.clone(),
            runtime: profile.runtime.clone(),
            model: profile.model.clone(),
            slot_count: profile.slot_count,
            inner: Mutex::new(inner),
            dirty: AtomicBool::new(false),
            loaded_from_store: true,
        }
    }

    fn lock(&self) -> MutexGuard<'_, CalibInner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn loaded_from_store(&self) -> bool {
        self.loaded_from_store
    }

    /// True when observations changed the profile since the last save.
    pub fn take_dirty(&self) -> bool {
        self.dirty.swap(false, Ordering::Relaxed)
    }

    // -- dispatch registry ----------------------------------------------------

    /// Register a dispatch (reservation time) and mint its ticket. The
    /// registry is what "how many streams are active" counts — reservations
    /// count, because a burst prices each other in before any bytes move.
    pub fn register_dispatch(&self, est_total_ms: f64, est_prefill_ms: f64) -> DispatchTicket {
        let mut g = self.lock();
        let id = g.next_id;
        g.next_id += 1;
        g.registry.push(RegistryEntry {
            id,
            started: Instant::now(),
            est_prefill_ms,
        });
        DispatchTicket { start: Instant::now(), id, est_total_ms, est_prefill_ms }
    }

    /// Remove a finished dispatch (idempotent: unknown ids are ignored).
    pub fn unregister_dispatch(&self, ticket: &DispatchTicket) {
        let mut g = self.lock();
        g.registry.retain(|e| e.id != ticket.id);
    }

    /// Server-wide count of dispatches estimated to be prefilling.
    pub fn active_prefills(&self) -> usize {
        let g = self.lock();
        g.registry.iter().filter(|e| e.started.elapsed().as_secs_f64() * 1000.0 < e.est_prefill_ms).count()
    }

    /// Server-wide count of dispatches past their estimated prefill —
    /// the decode-load signal. `8 requests in prefill` and `8 requests
    /// decoding` have radically different consequences; never conflate.
    pub fn active_decodes(&self) -> usize {
        let g = self.lock();
        g.registry.iter().filter(|e| e.started.elapsed().as_secs_f64() * 1000.0 >= e.est_prefill_ms).count()
    }

    // -- learning ----------------------------------------------------------------

    /// Feed one prefill observation (response headers → first token, ms)
    /// that ran under `bucket(peers + 1)` concurrent prefills, where `peers`
    /// excludes the observed dispatch itself. The SOLO regression learns
    /// only from unloaded (bucket-1) observations — feeding 8-way-parallel
    /// prefills into it is how a contention penalty becomes "the rate"
    /// (the exact one-rate-for-all-loads mistake this module exists to
    /// remove). Loaded observations update their bucket's contention
    /// multiplier against the current solo rate instead.
    pub fn observe_prefill(&self, input_tokens: u64, worker_first_token_ms: f64, ticket: &DispatchTicket) {
        let toks = input_tokens.max(1) as f64;
        let mut g = self.lock();
        let peers = g
            .registry
            .iter()
            .filter(|e| e.id != ticket.id && e.started.elapsed().as_secs_f64() * 1000.0 < e.est_prefill_ms)
            .count();
        let b = bucket_index(peers + 1);
        if peers == 0 {
            g.prefill.observe(toks, worker_first_token_ms);
        }
        // Contention vs the *solo* rate. Clamp guards a transient regression
        // jump from poisoning the multiplier.
        let ms_per_tok = worker_first_token_ms / toks;
        if g.prefill.rate > 0.1 {
            let mult_obs = (ms_per_tok / g.prefill.rate).clamp(0.25, 16.0);
            g.contention[b] = ewma_step(g.contention[b], mult_obs, CONTENTION_ALPHA).clamp(0.5, 16.0);
            g.contention_samples[b] += 1;
        }
        self.dirty.store(true, Ordering::Relaxed);
    }

    /// Feed one decode observation: `decode_ms` spent producing
    /// `output_tokens` under `bucket(peers + 1)` concurrent decode streams
    /// (`peers` excludes the observed dispatch itself).
    pub fn observe_decode(&self, output_tokens: u64, decode_ms: f64, ticket: &DispatchTicket) {
        let toks = output_tokens.max(1) as f64;
        let obs = (decode_ms / toks).clamp(1.0, 2_000.0);
        let mut g = self.lock();
        let peers = g
            .registry
            .iter()
            .filter(|e| e.id != ticket.id && e.started.elapsed().as_secs_f64() * 1000.0 >= e.est_prefill_ms)
            .count();
        let b = bucket_index(peers + 1);
        g.decode[b] = ewma_step(g.decode[b], obs, DECODE_ALPHA);
        g.decode_samples[b] += 1;
        self.dirty.store(true, Ordering::Relaxed);
    }

    // -- prediction -----------------------------------------------------------------

    /// Solo prefill regression: `(fixed_ms, ms_per_token)`.
    pub fn prefill_base(&self) -> (f64, f64) {
        let g = self.lock();
        (g.prefill.fixed, g.prefill.rate)
    }

    /// Prefill ms for `input_tokens` of which `reusable_tokens` are cached,
    /// under `active_prefills` concurrent prefills *including this one*.
    /// Applies the same measured-fallback chain as [`Self::contention_for`].
    pub fn prefill_ms(&self, input_tokens: u64, reusable_tokens: u64, active_prefills: usize) -> f64 {
        let g = self.lock();
        let mult = contention_locked(&g, active_prefills);
        g.prefill.fixed
            + input_tokens.saturating_sub(reusable_tokens) as f64 * g.prefill.rate * mult
    }

    /// Decode ms/token under `active_streams` concurrent decode streams
    /// including this one, with the measured-bucket fallback chain: exact
    /// bucket → nearest *higher* bucket (over-pricing loaded queues is the
    /// safe direction — 4B.2's failure was under-pricing) → nearest lower →
    /// physical seed.
    pub fn decode_rate_for(&self, active_streams: usize) -> f64 {
        let g = self.lock();
        decode_locked(&g, active_streams)
    }

    /// Contention multiplier for `active_prefills` concurrent prefills
    /// (including this one); same fallback chain as [`Self::decode_rate_for`].
    pub fn contention_for(&self, active_prefills: usize) -> f64 {
        let g = self.lock();
        contention_locked(&g, active_prefills)
    }

    /// Solo-rate decode ms/token — the "fastest honest path" number the
    /// admission controller and legacy formulas price with. Strictly solo:
    /// the measured bucket-1 rate, or the physical seed if bucket 1 has no
    /// observations yet. Loaded buckets never stand in for solo — that
    /// substitution would inflate every admission decision.
    pub fn decode_rate_solo(&self) -> f64 {
        let g = self.lock();
        if g.decode_samples[0] > 0 {
            g.decode[0]
        } else {
            g.seed_decode_ms_per_tok
        }
    }

    /// The single CostModel number admission control budgets with: this
    /// request's full service time on an otherwise idle server (solo
    /// prefill regression + solo decode bucket). Router and admission MUST
    /// consume the same model — the 4B.2 architecture kept separate
    /// "router is cheap / admission is expensive" opinions and they fought.
    pub fn solo_service_ms(&self, input_tokens: u64, output_tokens: u64) -> f64 {
        self.prefill_ms(input_tokens, 0, 1) + self.decode_rate_solo() * output_tokens as f64
    }

    /// Export the current state as a persistable profile.
    pub fn snapshot_profile(&self) -> CalibrationProfile {
        let g = self.lock();
        CalibrationProfile {
            schema_version: SCHEMA_VERSION,
            bucket_labels: BUCKET_LABELS.iter().map(|s| s.to_string()).collect::<Vec<_>>()
                .try_into()
                .unwrap_or_default(),
            machine: self.machine.clone(),
            runtime: self.runtime.clone(),
            model: self.model.clone(),
            slot_count: self.slot_count,
            updated_at_epoch_s: epoch_s(),
            prefill: PrefillCalibration {
                fixed_ms: g.prefill.fixed,
                ms_per_token: g.prefill.rate,
                samples: g.prefill.samples,
                contention_multiplier: g.contention,
                contention_samples: g.contention_samples,
            },
            decode: DecodeCalibration {
                ms_per_token: g.decode,
                samples: g.decode_samples,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defaults() -> ServiceDefaults {
        ServiceDefaults { fixed_ms: 250.0, prefill_ms_per_tok: 6.2, decode_ms_per_tok: 25.0 }
    }

    fn cal() -> RuntimeCalibration {
        RuntimeCalibration::from_seeds("apple-m5", "llama-cpp", "qwen-9b-q4", 8, defaults())
    }

    #[test]
    fn bucket_mapping() {
        assert_eq!(bucket_label(1), "1");
        assert_eq!(bucket_label(2), "2");
        assert_eq!(bucket_label(3), "3-4");
        assert_eq!(bucket_label(4), "3-4");
        assert_eq!(bucket_label(5), "5-8");
        assert_eq!(bucket_label(8), "5-8");
        assert_eq!(bucket_label(9), "5-8", "saturates above the top bucket");
        assert_eq!(bucket_label(0), "1", "a lone arrival is bucket 1");
    }

    #[test]
    fn slugify_makes_safe_paths() {
        assert_eq!(slugify("ggml-org/Qwen3.5-9B-Q4_K_M.gguf"), "ggml-org-qwen3.5-9b-q4_k_m.gguf");
        assert_eq!(slugify("  ///  "), "model");
        assert_eq!(slugify("mlx-community/Qwen3.5-9B-4bit"), "mlx-community-qwen3.5-9b-4bit");
    }

    #[test]
    fn store_roundtrip_and_missing_file() {
        let dir = std::env::temp_dir().join(format!("calib-test-{}", std::process::id()));
        let key = CalibrationKey::new("apple-m5", "llama-cpp", "Qwen/Qwen3.5-9B-Q4", 8);
        let store = CalibrationStore::new(&dir, &key);
        assert!(store.load().is_none(), "no file yet");

        let c = cal();
        let t = c.register_dispatch(1000.0, 500.0);
        c.observe_decode(256, 6400.0, &t); // 25 ms/tok bucket 1 (no peers)
        c.unregister_dispatch(&t);
        store.save(&c.snapshot_profile()).unwrap();

        let loaded = store.load().expect("profile roundtrips");
        assert_eq!(loaded.schema_version, SCHEMA_VERSION);
        assert_eq!(loaded.decode.samples, [1, 0, 0, 0]);
        // 25 (seed) + one EWMA step toward 25 = exactly 25.
        assert!((loaded.decode.ms_per_token[0] - 25.0).abs() < 0.01);
        assert_eq!(loaded.model, "qwen-9b-q4", "the profile records the model it calibrated");
        assert_eq!(loaded.bucket_labels, ["1", "2", "3-4", "5-8"]);

        // Corrupt file → treated as absent, never a panic.
        std::fs::write(store.path(), "{not json").unwrap();
        assert!(store.load().is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn decode_buckets_learn_and_fall_back_upward() {
        let c = cal();
        // One loaded observation in bucket 5-8: 8 peers + self = 9 streams
        // is bucket 5-8; simulate by registering 8 long-decode dispatches.
        let mut tickets = Vec::new();
        for _ in 0..8 {
            tickets.push(c.register_dispatch(600_000.0, 1.0)); // will read "decoding"
        }
        // The registry classifies by elapsed-vs-estimated-prefill; give the
        // peers their 1 ms so they count as decode streams.
        std::thread::sleep(std::time::Duration::from_millis(5));
        let t = c.register_dispatch(600_000.0, 1.0);
        // The registry classifies SELF by elapsed-vs-est-prefill (just
        // registered → still "prefilling"), while the OBSERVATION buckets
        // by peers+1 — 8 decode peers means this decode ran at load 9.
        assert_eq!(c.active_decodes(), 8);
        assert_eq!(c.active_prefills(), 1);
        c.observe_decode(256, 256.0 * 250.0, &t); // 250 ms/tok under 9 streams
        for tk in &tickets {
            c.unregister_dispatch(tk);
        }
        c.unregister_dispatch(&t);

        let p = c.snapshot_profile();
        assert_eq!(p.decode.samples, [0, 0, 0, 1]);
        // One EWMA step from the 25 seed toward 250: 25 + 0.25 × 225 = 81.25.
        assert!((p.decode.ms_per_token[3] - 81.25).abs() < 0.1);

        // Prediction: bucket 5-8 measured; buckets below fall back UP to it
        // (over-pricing beats the 4B.2 under-pricing failure).
        assert!((c.decode_rate_for(8) - 81.25).abs() < 0.1);
        assert!((c.decode_rate_for(2) - 81.25).abs() < 0.1);
        assert!((c.decode_rate_for(1) - 81.25).abs() < 0.1);
        // ...but SOLO pricing falls back to the physical seed: an unloaded
        // request really does decode at ~25 ms/tok.
        assert!((c.decode_rate_solo() - 25.0).abs() < 0.001);
        assert!(c.take_dirty(), "observations mark the profile dirty");
        assert!(!c.take_dirty(), "dirty flag is take-once");
    }

    #[test]
    fn prefill_contention_multiplier_learns() {
        let c = cal();
        // Train the solo regression: 100ms + 6ms/tok, varied sizes.
        let mut t = c.register_dispatch(1000.0, 500.0);
        for i in 0..60 {
            let toks = 500 + (i % 7) * 350;
            c.unregister_dispatch(&t);
            t = c.register_dispatch(1.0, f64::MAX); // stays "prefilling"
            c.observe_prefill(toks, 100.0 + 6.0 * toks as f64, &t);
        }
        assert!((c.prefill_base().1 - 6.0).abs() < 0.5, "solo rate learned: {}", c.prefill_base().1);

        // Now 7 peer prefills active → this one is bucket 5-8 (peers + self).
        let mut peers = Vec::new();
        for _ in 0..7 {
            peers.push(c.register_dispatch(1.0, f64::MAX));
        }
        c.unregister_dispatch(&t); // the training dispatch leaves the server
        let tk = c.register_dispatch(1.0, f64::MAX);
        assert_eq!(c.active_prefills(), 8);
        c.observe_prefill(8400, 8400.0 * 15.0, &tk); // 15 ms/tok vs ~6 solo
        let mult = c.contention_for(8);
        assert!(mult > 1.2, "contention multiplier learned upward: {mult}");
        // Prediction reflects it: 8-way prefill prices above the solo rate.
        let cold8 = c.prefill_ms(8400, 0, 8);
        assert!(cold8 > 8400.0 * 6.0 * 1.15, "bucketed prefill {cold8} reflects contention");
        // A solo arrival on an idle server prices at ~solo.
        let solo = c.prefill_ms(8400, 0, 1);
        assert!(solo < 8400.0 * 6.0 * 1.2, "solo prefill {solo} near base rate");
    }

    #[test]
    fn registry_counts_split_by_phase() {
        let c = cal();
        // Two fresh prefills, one dispatch old enough to be "decoding".
        let _a = c.register_dispatch(10_000.0, 9_000.0);
        let _b = c.register_dispatch(10_000.0, 9_000.0);
        let old = c.register_dispatch(60_000.0, 1.0);
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert_eq!(c.active_prefills(), 2, "a and b are inside their prefill window");
        assert_eq!(c.active_decodes(), 1, "old is past its tiny prefill estimate");
        c.unregister_dispatch(&old);
        assert_eq!(c.active_decodes(), 0);
        // Idempotent unregister.
        c.unregister_dispatch(&old);
        assert_eq!(c.active_prefills(), 2);
    }

    #[test]
    fn from_profile_adopts_state_and_flags_origin() {
        let dir = std::env::temp_dir().join(format!("calib-test2-{}", std::process::id()));
        let key = CalibrationKey::new("apple-m5", "llama-cpp", "m", 8);
        let store = CalibrationStore::new(&dir, &key);
        let src = cal();
        let mut tickets = Vec::new();
        for _ in 0..8 {
            tickets.push(src.register_dispatch(1.0, 1.0));
        }
        std::thread::sleep(std::time::Duration::from_millis(5)); // peers become "decoding"
        let t = src.register_dispatch(1.0, 1.0);
        src.observe_decode(64, 64.0 * 200.0, &t); // 200 ms/tok at 9 streams
        store.save(&src.snapshot_profile()).unwrap();
        for tk in &tickets {
            src.unregister_dispatch(tk);
        }
        src.unregister_dispatch(&t);

        let p = store.load().unwrap();
        let adopted = RuntimeCalibration::from_profile(&p, defaults());
        assert!(adopted.loaded_from_store());
        // One EWMA step from the 25 seed toward 200: 25 + 0.25 × 175 = 68.75.
        assert!((adopted.decode_rate_for(8) - 68.75).abs() < 0.01,
            "adopted profile drives bucket predictions");
        std::fs::remove_dir_all(&dir).ok();
    }
}
