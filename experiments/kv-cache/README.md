# kv-cache

KV pressure: push context until cache thrashes (input 8K/16K, concurrency up); watch memory footprint and ITL degradation. Record where admission/eviction starts hurting.

Hypothesis, method, kept results and writeup live here as the experiment lands.
See docs/benchmark-methodology.md for measurement rules.
