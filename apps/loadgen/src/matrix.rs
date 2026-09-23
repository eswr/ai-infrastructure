//! Experiment-matrix manifests: one file describing a whole study.
//!
//! A matrix is shared defaults plus a list of named runs (and optional
//! warmup runs) executed sequentially by the runner, with a cooldown
//! between runs so queues drain and thermals settle. Every run lists only
//! the Scenario fields it varies; the rest fall through run → defaults →
//! the built-in baseline. Warmup runs prime the model (weights page-in,
//! JIT) and are recorded but flagged so analysis can exclude them.

use anyhow::{Context, Result};
use inference_lab_workload::{Arrival, Scenario};
use serde::{Deserialize, Serialize};

pub fn parse_arrival(s: &str) -> Result<Arrival> {
    let s = s.trim();
    match s {
        "burst" => Ok(Arrival::Burst),
        _ if s.starts_with("constant:") => Ok(Arrival::Constant {
            interval_s: s["constant:".len()..].parse().context("bad constant interval")?,
        }),
        _ if s.starts_with("poisson:") => Ok(Arrival::Poisson {
            mean_interval_s: s["poisson:".len()..].parse().context("bad poisson interval")?,
        }),
        _ => anyhow::bail!("arrival must be burst | constant:<s> | poisson:<s>, got {s:?}"),
    }
}

/// Arrival rendered back to its manifest spelling (for artifacts).
pub fn format_arrival(a: &Arrival) -> String {
    match a {
        Arrival::Burst => "burst".into(),
        Arrival::Constant { interval_s } => format!("constant:{interval_s}"),
        Arrival::Poisson { mean_interval_s } => format!("poisson:{mean_interval_s}"),
    }
}

/// One entry of a matrix: any Scenario field may be set; unset fields
/// inherit from the lower layer.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunSpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub concurrency: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_requests: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shared_prefix_tokens: Option<usize>,
    /// burst | constant:<interval_s> | poisson:<mean_interval_s>
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arrival: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_s: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

impl RunSpec {
    /// Layer this spec over `defaults`, applied onto `base`: the first
    /// layer that sets a field wins.
    pub fn resolve(&self, defaults: &RunSpec, base: &Scenario, fallback_name: &str) -> Result<Scenario> {
        let mut s = base.clone();
        if let Some(v) = self.endpoint.as_ref().or(defaults.endpoint.as_ref()) {
            s.endpoint = v.clone();
        }
        if let Some(v) = self.model.as_ref().or(defaults.model.as_ref()) {
            s.model = v.clone();
        }
        if let Some(v) = self.concurrency.or(defaults.concurrency) {
            s.concurrency = v;
        }
        if let Some(v) = self.num_requests.or(defaults.num_requests) {
            s.num_requests = v;
        }
        if let Some(v) = self.input_tokens.or(defaults.input_tokens) {
            s.input_tokens = v;
        }
        if let Some(v) = self.output_tokens.or(defaults.output_tokens) {
            s.output_tokens = v;
        }
        if let Some(v) = self.stream.or(defaults.stream) {
            s.stream = v;
        }
        if let Some(v) = self.shared_prefix_tokens.or(defaults.shared_prefix_tokens) {
            s.shared_prefix_tokens = v;
        }
        if let Some(v) = self.arrival.as_ref().or(defaults.arrival.as_ref()) {
            s.arrival = parse_arrival(v)?;
        }
        if let Some(v) = self.timeout_s.or(defaults.timeout_s) {
            s.timeout_s = v;
        }
        if let Some(v) = self.seed.or(defaults.seed) {
            s.seed = v;
        }
        s.name = self
            .name
            .clone()
            .unwrap_or_else(|| fallback_name.to_string());
        Ok(s)
    }
}

/// Parsed matrix manifest (see benchmarks/matrices/).
#[derive(Debug, Clone, Deserialize)]
pub struct Matrix {
    /// Study slug; names the results directory.
    pub study: String,
    #[serde(default)]
    pub description: String,
    /// Quiet seconds between runs.
    #[serde(default = "default_cooldown")]
    pub cooldown_s: f64,
    #[serde(default)]
    pub defaults: RunSpec,
    #[serde(default)]
    pub warmup: Vec<RunSpec>,
    pub runs: Vec<RunSpec>,
}

fn default_cooldown() -> f64 {
    3.0
}

impl Matrix {
    pub fn load_toml(path: &str) -> Result<Matrix> {
        let raw = std::fs::read_to_string(path)?;
        let m: Matrix = toml::from_str(&raw).with_context(|| format!("parsing matrix {path}"))?;
        m.validate()?;
        Ok(m)
    }

    fn validate(&self) -> Result<()> {
        anyhow::ensure!(!self.runs.is_empty(), "matrix has no [[runs]]");
        anyhow::ensure!(self.cooldown_s >= 0.0, "cooldown_s must be >= 0");
        let mut seen = std::collections::HashSet::new();
        for (i, r) in self.runs.iter().enumerate() {
            let name = r.name.clone().unwrap_or_else(|| format!("run-{i}"));
            anyhow::ensure!(
                seen.insert(name.clone()),
                "duplicate run name {name:?}; names must be unique (they name artifacts)"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> Scenario {
        Scenario {
            name: "base".into(),
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

    #[test]
    fn run_overrides_win_over_defaults() {
        let defaults = RunSpec {
            output_tokens: Some(256),
            endpoint: Some("http://default".into()),
            ..Default::default()
        };
        let run = RunSpec {
            name: Some("p128-c4".into()),
            input_tokens: Some(128),
            concurrency: Some(4),
            endpoint: Some("http://run".into()),
            ..Default::default()
        };
        let s = run.resolve(&defaults, &base(), "unused").unwrap();
        assert_eq!(s.name, "p128-c4");
        assert_eq!(s.endpoint, "http://run"); // run layer wins
        assert_eq!(s.output_tokens, 256); // falls through to defaults
        assert_eq!(s.concurrency, 4);
        assert_eq!(s.input_tokens, 128);
    }

    #[test]
    fn arrival_strings_parse_both_layers() {
        let defaults = RunSpec {
            arrival: Some("poisson:2.5".into()),
            ..Default::default()
        };
        let run = RunSpec { ..Default::default() };
        let s = run.resolve(&defaults, &base(), "w").unwrap();
        assert!(matches!(s.arrival, Arrival::Poisson { mean_interval_s } if mean_interval_s == 2.5));

        let run = RunSpec {
            arrival: Some("constant:0.5".into()),
            ..Default::default()
        };
        let s = run.resolve(&defaults, &base(), "w").unwrap();
        assert!(matches!(s.arrival, Arrival::Constant { interval_s } if interval_s == 0.5));
    }

    #[test]
    fn matrix_parses_and_rejects_duplicate_names() {
        let raw = r#"
            study = "x"
            [defaults]
            endpoint = "http://e"
            model = "m"
            [[runs]]
            name = "a"
            concurrency = 1
            [[runs]]
            name = "a"
            concurrency = 2
        "#;
        assert!(toml::from_str::<Matrix>(raw).unwrap().validate().is_err());

        let raw = raw.replace("\"a\", concurrency = 2", "\"b\", concurrency = 2");
        let m: Matrix = toml::from_str(&raw).unwrap();
        assert_eq!(m.runs.len(), 2);
        assert_eq!(m.cooldown_s, 3.0); // default
        assert!(m.warmup.is_empty());
    }

    #[test]
    fn fallback_name_used_when_unnamed() {
        let run = RunSpec {
            concurrency: Some(2),
            ..Default::default()
        };
        let s = run.resolve(&RunSpec::default(), &base(), "warmup-0").unwrap();
        assert_eq!(s.name, "warmup-0");
    }
}
