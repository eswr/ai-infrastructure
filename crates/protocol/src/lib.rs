//! Wire types for the OpenAI-compatible chat API plus the worker
//! advertisement format used by the gateway/router.
//!
//! These types are deliberately permissive (unknown fields are ignored) so
//! the same structs work against vLLM-Metal, MLX-LM, llama.cpp and the mock
//! worker, all of which drift slightly from the OpenAI schema.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Chat completions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Bumped by loadgen so servers that support it report token usage in the
    /// final streamed chunk.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub include_usage: bool,
    /// Pin the request to one llama.cpp slot (`-np` parallel cell, its own
    /// KV cache). Set by the gateway's slot router after it picks a slot,
    /// and by loadgen warmups that populate a specific slot's cache.
    /// `None` serializes nothing — llama.cpp then picks the slot itself
    /// (LCP matching), which is the `native` baseline of the routing study.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id_slot: Option<u32>,
    /// Ask llama.cpp to reuse its cached prefix for this prompt. Set
    /// alongside `id_slot` by the slot router; llama.cpp 0.4.1 defaults it
    /// on, but the router's cache economics assume reuse actually happens,
    /// so it pins the behavior explicitly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_prompt: Option<bool>,
    /// Lab extension (phase 4C): requested service class — `interactive`,
    /// `standard` or `batch`. The gateway's scheduler consumes it (class
    /// priority in admission queue grants, class-labeled telemetry);
    /// ordinary OpenAI servers ignore the unknown field entirely.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_class: Option<String>,
    /// Lab extension (phase 6A): datacenter placement requirements for a
    /// heterogeneous accelerator fleet. When absent the gateway performs NO
    /// placement filtering — requests behave exactly as on the homogeneous
    /// lab fleet.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placement: Option<PlacementRequirements>,
}

/// What a request demands from the accelerator it lands on (phase 6A).
/// `model_memory_gb` overrides the registry estimate when the caller knows
/// better; otherwise the gateway estimates
/// `weights + kv_cache(context) + runtime_overhead` with a safety margin
/// and rejects impossible placements BEFORE dispatch.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PlacementRequirements {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_memory_gb: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tensor_parallel_size: Option<u32>,
    /// e.g. `nvidia-h100` | `nvidia-l40s` | `nvidia-a10` | `apple-m5`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accelerator: Option<String>,
}

fn default_max_tokens() -> u32 {
    128
}

fn default_temperature() -> f32 {
    1.0
}

impl ChatCompletionRequest {
    pub fn simple(model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            messages: vec![Message {
                role: "user".into(),
                content: prompt.into(),
            }],
            stream: false,
            placement: None,
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
            top_p: None,
            include_usage: false,
            id_slot: None,
            cache_prompt: None,
            service_class: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    #[serde(default)]
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    pub finish_reason: Option<String>,
}

/// One streamed chunk (`object: "chat.completion.chunk"`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    #[serde(default)]
    pub choices: Vec<ChunkChoice>,
    /// Present in the final chunk when the request asked for `include_usage`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Delta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Thinking models stream generated tokens here instead of `content`
    /// during the reasoning phase — vLLM/MLX dialect.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    /// llama.cpp's dialect for the same thing: Qwen3.5 thinking tokens
    /// arrive as `reasoning_content`, and missing this field silently
    /// zero-counts every llama.cpp token.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

/// Extract the generated text from a streamed chunk, if any. Reasoning
/// tokens count too — for latency measurement a token is a token. Both
/// reasoning dialects (`reasoning`, `reasoning_content`) count.
pub fn chunk_text(chunk: &ChatCompletionChunk) -> Option<&str> {
    chunk
        .choices
        .iter()
        .find_map(|c| {
            c.delta
                .content
                .as_deref()
                .or(c.delta.reasoning.as_deref())
                .or(c.delta.reasoning_content.as_deref())
        })
        .filter(|s| !s.is_empty())
}

// ---------------------------------------------------------------------------
// Worker advertisement (what each worker exposes for routing decisions)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct WorkerStatus {
    pub worker_id: String,
    pub model: String,
    /// `vllm-metal` | `mlx-lm` | `llama-cpp` | `mock`
    pub runtime: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    #[serde(default)]
    pub max_context: u64,
    pub active_requests: u64,
    pub waiting_requests: u64,
    /// 0.0–1.0
    #[serde(default)]
    pub kv_utilization: f32,
    #[serde(default)]
    pub prefixes_cached: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_p50_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_p95_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_tok_s: Option<f64>,
    /// 0.0–1.0
    #[serde(default)]
    pub memory_pressure: f32,
    pub healthy: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_stream_chunk_with_unknown_fields() {
        let raw = r#"{"id":"1","object":"chat.completion.chunk","created":1,"model":"m",
                      "choices":[{"index":0,"delta":{"content":"tok"},"finish_reason":null}],
                      "logprobs":null,"system_fingerprint":"fp"}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        assert_eq!(chunk_text(&chunk), Some("tok"));
    }

    #[test]
    fn parses_done_usage_chunk() {
        let raw = r#"{"id":"1","object":"chat.completion.chunk","created":1,"model":"m",
                      "choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        assert_eq!(chunk.usage.unwrap().completion_tokens, 5);
    }

    #[test]
    fn counts_both_reasoning_dialects() {
        // llama.cpp streams thinking models as reasoning_content; a chunk
        // with ONLY that field must still count as a token (this silently
        // zero-counted every llama.cpp token before the field existed).
        let raw = r#"{"id":"1","object":"chat.completion.chunk","created":1,"model":"m",
                      "choices":[{"index":0,"delta":{"reasoning_content":"Thinking"},"finish_reason":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        assert_eq!(chunk_text(&chunk), Some("Thinking"));

        // vLLM/MLX dialect.
        let raw = raw.replace("reasoning_content", "reasoning");
        let chunk: ChatCompletionChunk = serde_json::from_str(&raw).unwrap();
        assert_eq!(chunk_text(&chunk), Some("Thinking"));
    }

    #[test]
    fn slot_fields_serialize_only_when_set() {
        // Without slot routing the wire format must stay exactly what the
        // other runtimes (mlx/vllm) already parse: no id_slot key at all.
        let plain = ChatCompletionRequest::simple("m", "hi");
        let v = serde_json::to_value(&plain).unwrap();
        assert!(v.get("id_slot").is_none());
        assert!(v.get("cache_prompt").is_none());

        let pinned = serde_json::from_value::<ChatCompletionRequest>(serde_json::json!({
            "model": "m", "messages": [], "id_slot": 3, "cache_prompt": true
        }))
        .unwrap();
        assert_eq!(pinned.id_slot, Some(3));
        assert_eq!(pinned.cache_prompt, Some(true));
        let v = serde_json::to_value(&pinned).unwrap();
        assert_eq!(v["id_slot"], 3);
        assert_eq!(v["cache_prompt"], true);
    }
}
