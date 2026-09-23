//! Incremental server-sent-events (SSE) parsing.
//!
//! The loadgen feeds raw response bytes into [`SseParser::feed`] and receives
//! complete SSE messages back; partial events stay buffered across chunk
//! boundaries. Handles `\n\n`, `\r\n\r\n` and `\r\r` separators, multi-line
//! `data:` fields and comment lines, per the SSE spec.

#[derive(Debug, Clone, PartialEq)]
pub struct SseMessage {
    /// Value of the `event:` line, if present.
    pub event: Option<String>,
    /// All `data:` lines joined with `\n`.
    pub data: String,
}

#[derive(Debug, Default)]
pub struct SseParser {
    buf: String,
}

impl SseParser {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed raw bytes (any UTF-8-ish boundary is fine — lossy conversion is
    /// applied) and drain every complete event now available.
    pub fn feed(&mut self, bytes: &[u8]) -> Vec<SseMessage> {
        self.buf.push_str(&String::from_utf8_lossy(bytes));
        // Normalize line endings once per feed; buffers are small.
        if self.buf.contains('\r') {
            self.buf = self.buf.replace("\r\n", "\n").replace('\r', "\n");
        }

        let mut out = Vec::new();
        while let Some(pos) = self.buf.find("\n\n") {
            let event_block: String = self.buf.drain(..pos + 2).collect();
            if let Some(msg) = parse_event_block(&event_block) {
                out.push(msg);
            }
        }
        out
    }

    /// Bytes buffered awaiting a terminator (useful for debugging stalls).
    pub fn pending_len(&self) -> usize {
        self.buf.len()
    }
}

fn parse_event_block(block: &str) -> Option<SseMessage> {
    let mut event: Option<String> = None;
    let mut data_lines: Vec<&str> = Vec::new();

    for line in block.lines() {
        let line = line.strip_suffix('\n').unwrap_or(line);
        if line.is_empty() || line.starts_with(':') {
            continue; // separator remainder or comment / keepalive
        }
        let (field, value) = match line.split_once(':') {
            Some((f, v)) => (f, v.strip_prefix(' ').unwrap_or(v)),
            None => (line, ""),
        };
        match field {
            "data" => data_lines.push(value),
            "event" => event = Some(value.to_string()),
            _ => {} // id:, retry: — ignored
        }
    }

    if data_lines.is_empty() {
        return None;
    }
    Some(SseMessage {
        event,
        data: data_lines.join("\n"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_data_event() {
        let mut p = SseParser::new();
        let msgs = p.feed(b"data: hello\n\n");
        assert_eq!(msgs, vec![SseMessage { event: None, data: "hello".into() }]);
    }

    #[test]
    fn buffers_partial_events_across_feeds() {
        let mut p = SseParser::new();
        assert!(p.feed(b"data: {\"a\":1}").is_empty());
        assert!(p.feed(b"\n\n").len() == 1);
    }

    #[test]
    fn handles_crlf_and_multi_data_lines() {
        let mut p = SseParser::new();
        let msgs = p.feed(b"event: message\r\ndata: line1\r\ndata: line2\r\n\r\n");
        assert_eq!(
            msgs,
            vec![SseMessage { event: Some("message".into()), data: "line1\nline2".into() }]
        );
    }

    #[test]
    fn skips_comments_and_ignores_done_semantics() {
        let mut p = SseParser::new();
        let msgs = p.feed(b": keepalive\n\ndata: [DONE]\n\n");
        // Comment-only block produces nothing; [DONE] is just data for the caller.
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].data, "[DONE]");
    }
}
