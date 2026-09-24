//! Latency recording and aggregation for benchmark runs.
//!
//! A [`RequestRecord`] captures the client-side timeline of one request
//! (all timestamps relative to request start). [`Summary`] aggregates
//! records into the percentiles the lab reports: TTFT, ITL, TPOT, E2E and
//! throughput. Measurement happens on the load-generator's clock, which is
//! the latency the user actually experiences.

use serde::Serialize;

/// One measured request.
#[derive(Debug, Clone, Default)]
pub struct RequestRecord {
    pub id: usize,
    pub ok: bool,
    pub error: Option<String>,
    /// Scheduled release offset from run start (0 for burst arrivals).
    /// Completion times relative to a common origin need this: e2e alone
    /// only orders fairly when every request was released together.
    pub release_offset_ms: f64,
    /// Elapsed at the first generated token, if the request got that far.
    pub ttft_ms: Option<f64>,
    /// Elapsed at every generated token (including the first).
    pub token_ts_ms: Vec<f64>,
    /// Elapsed when the response completed.
    pub e2e_ms: Option<f64>,
    /// Prompt tokens, when the server reported usage (estimate otherwise).
    pub prompt_tokens: Option<u64>,
    /// Generated tokens counted client-side (≈ streamed content chunks).
    pub completion_tokens: u64,
}

impl RequestRecord {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            ..Default::default()
        }
    }

    pub fn fail(&mut self, err: impl Into<String>) {
        self.ok = false;
        self.error = Some(err.into());
    }

    /// Inter-token latencies: gaps between consecutive tokens.
    pub fn itl_ms(&self) -> Vec<f64> {
        self.token_ts_ms.windows(2).map(|w| w[1] - w[0]).collect()
    }

    /// This stream's worst gap between consecutive tokens — the stall a
    /// user would have sat through even when everything else looked fine.
    pub fn max_inter_token_gap_ms(&self) -> Option<f64> {
        self.itl_ms().iter().copied().reduce(f64::max)
    }

    /// When this request finished, relative to run start (not request
    /// start). The completion-time spread across streams is the
    /// "everyone arrived together, when did everyone get home" fairness
    /// measure; it is only defined on successful requests.
    pub fn completion_ms(&self) -> Option<f64> {
        self.e2e_ms.map(|e| self.release_offset_ms + e)
    }

    /// Mean time per output token over the decode phase:
    /// (e2e - ttft) / (tokens - 1), defined once ≥2 tokens and ttft exist.
    pub fn tpot_ms(&self) -> Option<f64> {
        let ttft = self.ttft_ms?;
        let e2e = self.e2e_ms?;
        let n = self.completion_tokens;
        if n < 2 || e2e < ttft {
            return None;
        }
        Some((e2e - ttft) / (n as f64 - 1.0))
    }

    pub fn output_tok_s(&self) -> Option<f64> {
        let e2e = self.e2e_ms?;
        if e2e <= 0.0 || self.completion_tokens == 0 {
            return None;
        }
        Some(self.completion_tokens as f64 / (e2e / 1000.0))
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Percentiles {
    pub mean: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
}

fn stats_of(values: &mut [f64]) -> Option<Percentiles> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(Percentiles {
        mean: values.iter().sum::<f64>() / values.len() as f64,
        p50: percentile(values, 0.50),
        p95: percentile(values, 0.95),
        p99: percentile(values, 0.99),
        max: values[values.len() - 1],
    })
}

/// Linear-interpolated percentile over an already-sorted slice.
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = p.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let frac = rank - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Slowest/fastest e2e ratio among successful requests. Only defined with
/// ≥2 successful requests and a positive fastest stream.
fn e2e_ok_ratio(records: &[RequestRecord]) -> Option<f64> {
    let vals = records
        .iter()
        .filter(|r| r.ok)
        .filter_map(|r| r.e2e_ms)
        .collect::<Vec<_>>();
    let min = vals.iter().copied().reduce(f64::min)?;
    let max = vals.iter().copied().reduce(f64::max)?;
    (vals.len() >= 2 && min > 0.0).then_some(max / min)
}

/// Aggregate over a run. `wall_time_s` spans first request start to last
/// completion, which is what throughput is divided by.
#[derive(Debug, Clone, Serialize)]
pub struct Summary {
    pub scenario: String,
    pub total_requests: usize,
    pub successful: usize,
    pub failed: usize,
    pub wall_time_s: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<Percentiles>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub itl_ms: Option<Percentiles>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tpot_ms: Option<Percentiles>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub e2e_ms: Option<Percentiles>,
    /// Stream-fairness view: distribution over requests of each stream's
    /// *worst* inter-token gap. Aggregate ITL can hide one starved stream
    /// behind many healthy ones; this cannot.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_max_stall_ms: Option<Percentiles>,
    /// Latest completion − earliest completion across successful streams
    /// (absolute time, release offsets included). Aggregate throughput is
    /// blind to this: all tokens arrive, but the last user waited seconds
    /// longer than the first.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_completion_spread_ms: Option<f64>,
    /// Slowest / fastest successful stream (by e2e). ≥1.0; large values
    /// mean some streams are second-class citizens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_slowest_fastest_ratio: Option<f64>,
    pub request_throughput: f64,
    pub output_tokens_total: u64,
    pub output_tok_s: f64,
    pub prompt_tokens_total: u64,
    pub input_tok_s: f64,
}

impl Summary {
    pub fn from_records(scenario: &str, records: &[RequestRecord], wall_time_s: f64) -> Self {
        let ok = records.iter().filter(|r| r.ok).count();
        let wall = if wall_time_s > 0.0 {
            wall_time_s
        } else {
            f64::EPSILON
        };

        let mut ttft = records.iter().filter_map(|r| r.ttft_ms).collect::<Vec<_>>();
        let mut e2e = records.iter().filter_map(|r| r.e2e_ms).collect::<Vec<_>>();
        let mut tpot = records
            .iter()
            .filter_map(|r| r.tpot_ms())
            .collect::<Vec<_>>();
        // ITL pools every inter-token gap across all requests.
        let mut itl = records.iter().flat_map(|r| r.itl_ms()).collect::<Vec<_>>();

        // Fairness: per-stream worst gap, plus the completion-time spread
        // and slowest/fastest ratio over successful streams.
        let mut stalls = records
            .iter()
            .filter(|r| r.ok)
            .filter_map(|r| r.max_inter_token_gap_ms())
            .collect::<Vec<_>>();
        let completions = records
            .iter()
            .filter(|r| r.ok)
            .filter_map(|r| r.completion_ms())
            .collect::<Vec<_>>();
        let spread = if completions.is_empty() {
            None
        } else {
            Some(
                completions.iter().copied().fold(f64::MIN, f64::max)
                    - completions.iter().copied().fold(f64::MAX, f64::min),
            )
        };
        let ratio = e2e_ok_ratio(records);

        let output_total: u64 = records.iter().map(|r| r.completion_tokens).sum();
        let prompt_total: u64 = records.iter().filter_map(|r| r.prompt_tokens).sum();

        Self {
            scenario: scenario.to_string(),
            total_requests: records.len(),
            successful: ok,
            failed: records.len() - ok,
            wall_time_s,
            ttft_ms: stats_of(&mut ttft),
            itl_ms: stats_of(&mut itl),
            tpot_ms: stats_of(&mut tpot),
            e2e_ms: stats_of(&mut e2e),
            stream_max_stall_ms: stats_of(&mut stalls),
            stream_completion_spread_ms: spread,
            stream_slowest_fastest_ratio: ratio,
            request_throughput: ok as f64 / wall,
            output_tokens_total: output_total,
            output_tok_s: output_total as f64 / wall,
            prompt_tokens_total: prompt_total,
            input_tok_s: prompt_total as f64 / wall,
        }
    }

    /// Human-readable report for the terminal.
    pub fn table(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();

        let pct = |v: &Option<Percentiles>| -> String {
            match v {
                Some(p) => format!("p50 {:>9.1}  p95 {:>9.1}  p99 {:>9.1}", p.p50, p.p95, p.p99),
                None => "n/a".to_string(),
            }
        };

        writeln!(s, "scenario      {}", self.scenario).unwrap();
        writeln!(
            s,
            "requests      {} total / {} ok / {} failed",
            self.total_requests, self.successful, self.failed
        )
        .unwrap();
        writeln!(s, "wall time     {:.2}s", self.wall_time_s).unwrap();
        writeln!(s, "ttft_ms       {}", pct(&self.ttft_ms)).unwrap();
        writeln!(s, "itl_ms        {}", pct(&self.itl_ms)).unwrap();
        writeln!(s, "tpot_ms       {}", pct(&self.tpot_ms)).unwrap();
        writeln!(s, "e2e_ms        {}", pct(&self.e2e_ms)).unwrap();
        if let Some(st) = &self.stream_max_stall_ms {
            writeln!(
                s,
                "stall_ms      p50 {:>9.1}  p95 {:>9.1}  max {:>9.1}  (per-stream worst gap)",
                st.p50, st.p95, st.max
            )
            .unwrap();
        }
        if let Some(sp) = self.stream_completion_spread_ms {
            write!(s, "spread        {:.2}s", sp / 1000.0).unwrap();
            if let Some(r) = self.stream_slowest_fastest_ratio {
                write!(s, "   slowest/fastest ×{r:.2}").unwrap();
            }
            writeln!(s).unwrap();
        }
        writeln!(s, "req/s         {:.2}", self.request_throughput).unwrap();
        writeln!(
            s,
            "output tok/s  {:.1}   ({} tokens)",
            self.output_tok_s, self.output_tokens_total
        )
        .unwrap();
        if self.prompt_tokens_total > 0 {
            writeln!(
                s,
                "input  tok/s  {:.1}   ({} tokens)",
                self.input_tok_s, self.prompt_tokens_total
            )
            .unwrap();
        }
        s
    }

    /// Per-request rows for spreadsheet-friendly analysis.
    pub fn records_csv(records: &[RequestRecord]) -> String {
        let mut s = String::from(
            "id,ok,error,ttft_ms,e2e_ms,completion_ms,completion_tokens,prompt_tokens,tpot_ms,output_tok_s,itl_p50_ms,itl_p95_ms,itl_p99_ms,max_gap_ms,release_offset_ms\n",
        );
        for r in records {
            let tpot = r.tpot_ms().map(|v| format!("{v:.2}")).unwrap_or_default();
            let tps = r
                .output_tok_s()
                .map(|v| format!("{v:.2}"))
                .unwrap_or_default();
            let err = r
                .error
                .as_deref()
                .map(|e| e.replace(',', ";"))
                .unwrap_or_default();
            let mut gaps = r.itl_ms();
            gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let ip = |p: f64| {
                if gaps.is_empty() {
                    String::new()
                } else {
                    format!("{:.2}", percentile(&gaps, p))
                }
            };
            let max_gap = r
                .max_inter_token_gap_ms()
                .map(|v| format!("{v:.2}"))
                .unwrap_or_default();
            let comp = r
                .completion_ms()
                .map(|v| format!("{v:.2}"))
                .unwrap_or_default();
            s.push_str(&format!(
                "{},{},{},{:.2},{:.2},{},{},{},{},{},{},{},{},{},{}\n",
                r.id,
                r.ok,
                err,
                r.ttft_ms.unwrap_or(f64::NAN),
                r.e2e_ms.unwrap_or(f64::NAN),
                comp,
                r.completion_tokens,
                r.prompt_tokens.map(|v| v.to_string()).unwrap_or_default(),
                tpot,
                tps,
                ip(0.50),
                ip(0.95),
                ip(0.99),
                max_gap,
                r.release_offset_ms,
            ));
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_interpolates() {
        let sorted = [1.0, 2.0, 3.0, 4.0];
        assert!((percentile(&sorted, 0.5) - 2.5).abs() < 1e-9);
        assert!((percentile(&sorted, 0.0) - 1.0).abs() < 1e-9);
        assert!((percentile(&sorted, 1.0) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn tpot_and_itl_derive_from_token_timeline() {
        let mut r = RequestRecord::new(0);
        r.ok = true;
        r.ttft_ms = Some(100.0);
        r.token_ts_ms = vec![100.0, 120.0, 140.0, 160.0];
        r.e2e_ms = Some(170.0);
        r.completion_tokens = 4;
        assert!((r.tpot_ms().unwrap() - 23.33).abs() < 0.01); // (170-100)/3
        assert_eq!(r.itl_ms(), vec![20.0, 20.0, 20.0]);
    }

    #[test]
    fn summary_counts_failures() {
        let mut a = RequestRecord::new(0);
        a.ok = true;
        a.e2e_ms = Some(100.0);
        a.completion_tokens = 10;
        let mut b = RequestRecord::new(1);
        b.fail("timeout");
        let s = Summary::from_records("t", &[a, b], 2.0);
        assert_eq!(s.successful, 1);
        assert_eq!(s.failed, 1);
        assert!((s.request_throughput - 0.5).abs() < 1e-9);
    }

    #[test]
    fn fairness_metrics_surface_the_starved_stream() {
        // Two streams start together; b stalls 5 s mid-decode and limps
        // home long after a. Pooled ITL p50 stays healthy — the per-stream
        // stall, spread and ratio are what expose this.
        let mut a = RequestRecord::new(0);
        a.ok = true;
        a.ttft_ms = Some(100.0);
        a.token_ts_ms = vec![100.0, 120.0, 140.0];
        a.e2e_ms = Some(150.0);
        let mut b = RequestRecord::new(1);
        b.ok = true;
        b.ttft_ms = Some(100.0);
        b.token_ts_ms = vec![100.0, 110.0, 5_110.0];
        b.e2e_ms = Some(5_120.0);

        let s = Summary::from_records("t", &[a, b], 6.0);
        let stall = s.stream_max_stall_ms.expect("stall stats present");
        assert!((stall.p50 - 2510.0).abs() < 1e-9); // median of {20, 5000}
        assert!((stall.max - 5000.0).abs() < 1e-9);
        assert!((s.stream_completion_spread_ms.unwrap() - 4_970.0).abs() < 1e-9);
        assert!((s.stream_slowest_fastest_ratio.unwrap() - 5_120.0 / 150.0).abs() < 1e-6);
    }

    #[test]
    fn completion_ms_includes_release_offset() {
        let mut r = RequestRecord::new(0);
        r.ok = true;
        r.release_offset_ms = 2_500.0;
        r.e2e_ms = Some(750.0);
        assert!((r.completion_ms().unwrap() - 3_250.0).abs() < 1e-9);

        // Spread must order by absolute completion, not e2e: a request
        // released later but served faster finishes earlier.
        let mut late_fast = RequestRecord::new(1);
        late_fast.ok = true;
        late_fast.release_offset_ms = 8_000.0;
        late_fast.e2e_ms = Some(100.0);
        let mut early_slow = RequestRecord::new(0);
        early_slow.ok = true;
        early_slow.e2e_ms = Some(9_000.0);
        let s = Summary::from_records("t", &[early_slow, late_fast], 10.1);
        assert!((s.stream_completion_spread_ms.unwrap() - 900.0).abs() < 1e-9);
    }
}
