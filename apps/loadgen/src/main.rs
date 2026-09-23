//! loadgen — OpenAI-compatible inference load generator.
//!
//! Two modes:
//! * **single run** (`--scenario file.toml` + CLI overrides) — one loadgen
//!   execution, artifacts under `benchmarks/results/`
//! * **matrix run** (`--matrix file.toml`) — a whole study: warmup runs,
//!   then measured runs executed sequentially with a cooldown between them.
//!   Each run lands in its own results directory alongside a `matrix.json`
//!   aggregating every run's parameters and summary for analysis.
//!
//! Emits requests per a scenario's arrival schedule (open-loop release,
//! closed-loop concurrency cap), records the client-side token timeline for
//! every request, and reports TTFT / ITL / TPOT / E2E / throughput
//! percentiles plus per-request JSON & CSV artifacts.
//!
//! Token counting uses the standard client-side approximation: one streamed
//! content chunk = one token. When the server honors `include_usage`, its
//! reported token counts are also captured.

mod matrix;

use anyhow::{Context, Result};
use clap::Parser;
use futures_util::StreamExt;
use inference_lab_metrics::{RequestRecord, Summary};
use inference_lab_protocol::{chunk_text, ChatCompletionChunk};
use inference_lab_streaming::SseParser;
use inference_lab_workload::{Arrival, Scenario};
use matrix::{format_arrival, parse_arrival, Matrix};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, Semaphore};

#[derive(Parser, Debug)]
#[command(
    name = "loadgen",
    version,
    about = "Load generator for OpenAI-compatible inference endpoints"
)]
struct Cli {
    /// Scenario TOML (see benchmarks/scenarios/). CLI flags override it.
    #[arg(long)]
    scenario: Option<String>,

    /// Matrix TOML (see benchmarks/matrices/): warmup + measured runs.
    #[arg(long, conflicts_with = "scenario")]
    matrix: Option<String>,

    #[arg(long)]
    endpoint: Option<String>,
    #[arg(long)]
    model: Option<String>,
    /// Max in-flight requests.
    #[arg(long)]
    concurrency: Option<usize>,
    #[arg(long)]
    num_requests: Option<usize>,
    /// Approximate prompt length in tokens.
    #[arg(long)]
    input_tokens: Option<usize>,
    /// max_tokens per request.
    #[arg(long, alias = "max-tokens")]
    output_tokens: Option<u32>,
    /// Prompt tokens shared verbatim across all requests (prefix-cache test).
    #[arg(long)]
    shared_prefix_tokens: Option<usize>,
    /// Streaming on/off; overrides the scenario file when given
    /// (e.g. `--stream false`).
    #[arg(long, value_parser = clap::builder::BoolishValueParser::new())]
    stream: Option<bool>,
    /// burst | constant:<interval_s> | poisson:<mean_interval_s>
    #[arg(long)]
    arrival: Option<String>,
    #[arg(long)]
    timeout_s: Option<f64>,
    #[arg(long, default_value = "benchmarks/results")]
    out_dir: String,
}

/// Baseline every scenario layering starts from (mirrors the ad-hoc run).
fn baseline_scenario() -> Scenario {
    Scenario {
        name: "ad-hoc".into(),
        endpoint: String::new(),
        model: String::new(),
        concurrency: 1,
        num_requests: 1,
        input_tokens: 128,
        output_tokens: 64,
        stream: true,
        shared_prefix_tokens: 0,
        arrival: Arrival::Burst,
        timeout_s: 300.0,
        seed: 42,
    }
}

fn build_scenario(cli: &Cli) -> Result<Scenario> {
    let mut s = match &cli.scenario {
        Some(path) => Scenario::load_toml(path).with_context(|| format!("loading {path}"))?,
        None => baseline_scenario(),
    };

    if let Some(v) = &cli.endpoint {
        s.endpoint = v.clone();
    }
    if let Some(v) = &cli.model {
        s.model = v.clone();
    }
    if let Some(v) = cli.concurrency {
        s.concurrency = v;
    }
    if let Some(v) = cli.num_requests {
        s.num_requests = v;
    }
    if let Some(v) = cli.input_tokens {
        s.input_tokens = v;
    }
    if let Some(v) = cli.output_tokens {
        s.output_tokens = v;
    }
    if let Some(v) = cli.shared_prefix_tokens {
        s.shared_prefix_tokens = v;
    }
    if let Some(v) = cli.arrival.as_deref() {
        s.arrival = parse_arrival(v)?;
    }
    if let Some(v) = cli.timeout_s {
        s.timeout_s = v;
    }
    if let Some(v) = cli.stream {
        s.stream = v;
    }
    anyhow::ensure!(!s.endpoint.is_empty(), "missing --endpoint (or a scenario file)");
    anyhow::ensure!(!s.model.is_empty(), "missing --model (or a scenario file)");
    Ok(s)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "loadgen=info".into()),
        )
        .init();

    let cli = Cli::parse();
    let client = reqwest::Client::builder()
        .pool_idle_timeout(Duration::from_secs(30))
        .build()?;

    if let Some(path) = &cli.matrix {
        return run_matrix(&client, path, &cli.out_dir).await;
    }

    let scenario = build_scenario(&cli)?;
    anyhow::ensure!(scenario.num_requests > 0, "num_requests must be > 0");
    anyhow::ensure!(scenario.concurrency > 0, "concurrency must be > 0");

    eprintln!(
        "loadgen → {}  model={}  stream={}  requests={}  concurrency={}  in≈{}tok  out={}tok",
        scenario.endpoint,
        scenario.model,
        scenario.stream,
        scenario.num_requests,
        scenario.concurrency,
        scenario.input_tokens,
        scenario.output_tokens
    );

    let (records, wall_s) = execute(&client, &scenario).await;
    let summary = Summary::from_records(&scenario.name, &records, wall_s);

    println!();
    println!("{}", summary.table());

    let ts = chrono_like_ts();
    std::fs::create_dir_all(&cli.out_dir)?;
    let base = format!("{}/{}-{}", cli.out_dir.trim_end_matches('/'), scenario.name, ts);
    write_run_artifacts(&base, &summary, &records)?;
    println!("wrote {base}.json");
    println!("wrote {base}.csv");

    if summary.failed > 0 {
        eprintln!("note: {} request(s) failed", summary.failed);
    }
    Ok(())
}

// ------------------------------------------------------------------ matrix --

async fn run_matrix(client: &reqwest::Client, path: &str, out_dir: &str) -> Result<()> {
    let m = Matrix::load_toml(path)?;

    // Resolve every spec up front so a bad manifest fails before any load.
    let base = baseline_scenario();
    let mut planned = Vec::with_capacity(m.warmup.len() + m.runs.len());
    for (i, spec) in m.warmup.iter().enumerate() {
        let s = spec.resolve(&m.defaults, &base, &format!("warmup-{i}"))?;
        planned.push((s, true));
    }
    for (i, spec) in m.runs.iter().enumerate() {
        let s = spec.resolve(&m.defaults, &base, &format!("run-{i}"))?;
        anyhow::ensure!(s.num_requests > 0, "run {:?} has num_requests = 0", s.name);
        anyhow::ensure!(s.concurrency > 0, "run {:?} has concurrency = 0", s.name);
        planned.push((s, false));
    }

    let ts = chrono_like_ts();
    let study_dir = format!("{}/{}-{}", out_dir.trim_end_matches('/'), m.study, ts);
    std::fs::create_dir_all(&study_dir)?;
    let total = planned.len();
    let first_measured = planned
        .iter()
        .find(|(_, warmup)| !warmup)
        .map(|(s, _)| (s.endpoint.clone(), s.model.clone()));
    eprintln!(
        "matrix {:?} — {} measured run(s), {} warmup → {study_dir}",
        m.study,
        m.runs.len(),
        m.warmup.len()
    );

    let study_start = Instant::now();
    let mut rows = Vec::with_capacity(total);

    for (idx, (scenario, is_warmup)) in planned.into_iter().enumerate() {
        eprintln!(
            "\n=== [{idx}/{total}] {} {:?}  in≈{}tok out={}tok c={} n={} arrival={} ===",
            if is_warmup { "warmup" } else { "run" },
            scenario.name,
            scenario.input_tokens,
            scenario.output_tokens,
            scenario.concurrency,
            scenario.num_requests,
            format_arrival(&scenario.arrival),
        );

        let (records, wall_s) = execute(client, &scenario).await;
        if is_warmup && records.iter().any(|r| !r.ok) {
            anyhow::bail!(
                "warmup run {:?} had failed requests — is {} up? aborting matrix",
                scenario.name,
                scenario.endpoint
            );
        }
        let summary = Summary::from_records(&scenario.name, &records, wall_s);
        println!("{}", summary.table());
        let base = format!("{study_dir}/{}", scenario.name);
        write_run_artifacts(&base, &summary, &records)?;

        rows.push(serde_json::json!({
            "name": scenario.name,
            "warmup": is_warmup,
            "params": {
                "endpoint": scenario.endpoint,
                "model": scenario.model,
                "concurrency": scenario.concurrency,
                "num_requests": scenario.num_requests,
                "input_tokens": scenario.input_tokens,
                "output_tokens": scenario.output_tokens,
                "stream": scenario.stream,
                "shared_prefix_tokens": scenario.shared_prefix_tokens,
                "arrival": format_arrival(&scenario.arrival),
                "seed": scenario.seed,
            },
            "summary": summary,
        }));
        if summary.failed > 0 && !is_warmup {
            eprintln!("note: {} request(s) failed in {:?}", summary.failed, scenario.name);
        }

        if idx + 1 < total && m.cooldown_s > 0.0 {
            eprintln!("cooldown {}s", m.cooldown_s as u64);
            tokio::time::sleep(Duration::from_secs_f64(m.cooldown_s)).await;
        }
    }

    let meta = serde_json::json!({
        "study": m.study,
        "description": m.description,
        "created": ts,
        "cooldown_s": m.cooldown_s,
        "manifest": path,
        "endpoint": first_measured.as_ref().map(|(e, _)| e.clone()),
        "model": first_measured.as_ref().map(|(_, mdl)| mdl.clone()),
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "loadgen_version": env!("CARGO_PKG_VERSION"),
        "total_wall_s": study_start.elapsed().as_secs_f64(),
    });
    let matrix_path = format!("{study_dir}/matrix.json");
    std::fs::write(
        &matrix_path,
        serde_json::to_vec_pretty(&serde_json::json!({ "meta": meta, "rows": rows }))?,
    )?;

    // One-glance result table for measured runs.
    println!("\n=== {} — summary ===", m.study);
    println!(
        "{:<14} {:>6} {:>10} {:>10} {:>10} {:>8} {:>8}",
        "run", "ok", "ttft p50", "itl p50", "e2e p50", "req/s", "tok/s"
    );
    for row in &rows {
        if row["warmup"].as_bool() == Some(true) {
            continue;
        }
        let name = row["name"].as_str().unwrap_or("?");
        let s = &row["summary"];
        let ok = format!(
            "{}/{}",
            s["successful"].as_u64().unwrap_or(0),
            s["total_requests"].as_u64().unwrap_or(0)
        );
        let p = |field: &str| {
            s[field]["p50"]
                .as_f64()
                .map(|v| format!("{v:>10.0}"))
                .unwrap_or_else(|| format!("{:>10}", "n/a"))
        };
        println!(
            "{:<14} {:>6} {} {} {} {:>8.2} {:>8.1}",
            name,
            ok,
            p("ttft_ms"),
            p("itl_ms"),
            p("e2e_ms"),
            s["request_throughput"].as_f64().unwrap_or(0.0),
            s["output_tok_s"].as_f64().unwrap_or(0.0),
        );
    }
    println!("wrote {matrix_path}");
    Ok(())
}

// ------------------------------------------------------------- execution --

/// Release all requests per the scenario's arrival schedule and wait for
/// completion. Returns records sorted by id plus wall time (first release
/// to last completion).
async fn execute(client: &reqwest::Client, scenario: &Scenario) -> (Vec<RequestRecord>, f64) {
    let offsets = scenario.arrival_offsets();
    let semaphore = Arc::new(Semaphore::new(scenario.concurrency));
    let records = Arc::new(Mutex::new(Vec::<RequestRecord>::with_capacity(
        scenario.num_requests,
    )));

    let run_start = Instant::now();
    let mut tasks = Vec::with_capacity(scenario.num_requests);

    for (i, target) in offsets.into_iter().enumerate() {
        // Open-loop release at the scheduled offset...
        let now = run_start.elapsed();
        if target > now {
            tokio::time::sleep(target - now).await;
        }
        // ...then wait for a concurrency slot before firing.
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        let client = client.clone();
        let scenario = scenario.clone();
        let records = records.clone();
        tasks.push(tokio::spawn(async move {
            let record = run_one(&client, &scenario, i).await;
            drop(permit);
            records.lock().await.push(record);
        }));
    }
    for t in tasks {
        let _ = t.await;
    }
    let wall = run_start.elapsed();

    let mut records = records.lock().await;
    records.sort_by_key(|r| r.id);
    (records.to_vec(), wall.as_secs_f64())
}

/// Write `<base>.json` (summary + per-request records) and `<base>.csv`.
fn write_run_artifacts(base: &str, summary: &Summary, records: &[RequestRecord]) -> Result<()> {
    let json_out = serde_json::json!({
        "summary": summary,
        "records": records.iter().map(|r| serde_json::json!({
            "id": r.id,
            "ok": r.ok,
            "error": r.error,
            "ttft_ms": r.ttft_ms,
            "e2e_ms": r.e2e_ms,
            "completion_tokens": r.completion_tokens,
            "prompt_tokens": r.prompt_tokens,
            "token_ts_ms": r.token_ts_ms,
        })).collect::<Vec<_>>(),
    });
    std::fs::write(format!("{base}.json"), serde_json::to_vec_pretty(&json_out)?)?;
    std::fs::write(format!("{base}.csv"), Summary::records_csv(records))?;
    Ok(())
}

/// "%Y%m%d-%H%M%S" without pulling in chrono.
fn chrono_like_ts() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Civil-time conversion (days→y/m/d) after Howard Hinnant's algorithm.
    let days = now / 86400;
    let secs = now % 86400;
    let (h, m, s) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mth = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mth <= 2 { y + 1 } else { y };
    format!("{y:04}{mth:02}{d:02}-{h:02}{m:02}{s:02}")
}

async fn run_one(client: &reqwest::Client, scenario: &Scenario, id: usize) -> RequestRecord {
    let mut record = RequestRecord::new(id);
    let t0 = Instant::now();
    record.prompt_tokens = Some(estimate_prompt_tokens(scenario));

    let url = format!("{}/v1/chat/completions", scenario.endpoint.trim_end_matches('/'));
    let req = scenario.request(id);

    let resp = match client
        .post(&url)
        .json(&req)
        .timeout(Duration::from_secs_f64(scenario.timeout_s))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            record.fail(format!("send: {e}"));
            return record;
        }
    };

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        record.fail(format!("http {status}: {}", body.chars().take(200).collect::<String>()));
        return record;
    }

    if !scenario.stream {
        match resp.json::<serde_json::Value>().await {
            Ok(v) => {
                if let Some(usage) = v.get("usage") {
                    let pt = usage.get("prompt_tokens").and_then(|x| x.as_u64());
                    let ct = usage.get("completion_tokens").and_then(|x| x.as_u64());
                    if let Some(pt) = pt {
                        record.prompt_tokens = Some(pt);
                    }
                    if let Some(ct) = ct {
                        record.completion_tokens = ct;
                    }
                }
                record.e2e_ms = Some(t0.elapsed().as_secs_f64() * 1000.0);
                record.ok = true;
            }
            Err(e) => record.fail(format!("decode body: {e}")),
        }
        return record;
    }

    // Streaming: parse SSE, timestamp every content chunk.
    let mut parser = SseParser::new();
    let mut stream = resp.bytes_stream();
    loop {
        match stream.next().await {
            Some(Ok(bytes)) => {
                for msg in parser.feed(&bytes) {
                    if msg.data == "[DONE]" {
                        continue;
                    }
                    let Ok(chunk) = serde_json::from_str::<ChatCompletionChunk>(&msg.data) else {
                        continue; // keepalives / non-chat events
                    };
                    if let (Some(usage), true) = (&chunk.usage, chunk.choices.is_empty()) {
                        // Final usage chunk: server-truth token counts.
                        if usage.prompt_tokens > 0 {
                            record.prompt_tokens = Some(usage.prompt_tokens);
                        }
                        if usage.completion_tokens > 0 {
                            record.completion_tokens = usage.completion_tokens;
                        }
                    }
                    if chunk_text(&chunk).is_some() {
                        let ms = t0.elapsed().as_secs_f64() * 1000.0;
                        if record.ttft_ms.is_none() {
                            record.ttft_ms = Some(ms);
                        }
                        record.token_ts_ms.push(ms);
                        // While streaming, the chunk count is our live estimate;
                        // a usage chunk at the end can only raise it.
                        let counted = record.token_ts_ms.len() as u64;
                        record.completion_tokens = record.completion_tokens.max(counted);
                    }
                }
            }
            Some(Err(e)) => {
                record.fail(format!("stream: {e}"));
                record.e2e_ms = Some(t0.elapsed().as_secs_f64() * 1000.0);
                return record;
            }
            None => break,
        }
    }

    record.e2e_ms = Some(t0.elapsed().as_secs_f64() * 1000.0);
    record.ok = record.error.is_none();
    record
}

fn estimate_prompt_tokens(s: &Scenario) -> u64 {
    (s.input_tokens.max(s.shared_prefix_tokens) + 64) as u64
}
