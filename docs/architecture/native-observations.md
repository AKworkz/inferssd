# Native observations

Every launch writes the engine's events to `native-events.jsonl` in its run
directory. `inferssd.telemetry.observations.NativeObservationSession` reads the
routes, layer costs and prefill summaries in it; the benchmark CI uses it to
record what each run did. Human-readable server logs and `INFERSSD_DEBUG` are
optional diagnostics.

The existing JSONL envelope is `v: 1`, a contiguous per-writer `seq`, and `ev`.
After `ssd.settings`, `ssd.controller.begin` declares `schema: 1`. The event keeps
the name it was given for the retired online controller, so recorded streams stay
readable. Each writer
belongs to one native runtime and is closed after its users have drained.

| Event | Fields read |
| --- | --- |
| `ssd.place.layer` | epoch, token, has_token, layer, decode, success, selected_experts, vram_hit, cache_hit, wall_us, wait_io_us, compute_us |
| `ssd.prefill.graph` | requested_tokens, configured_max, selected_ubatch |
| `ssd.prefill.layer` | layer, n_tokens, read_bytes |
| `ssd.prefill.copy` | bytes, read_us, passes, tokens |

Route arrays are captured from the completed native layer, not reconstructed
from text. Wall-time counters are already collected by the executor; enabling
this contract does not enable per-thread profiling or logits tracing. The
reader treats a token position at or below the previous one in the same epoch
as the start of a new run, so successive requests that reuse positions stay
apart. Prefill fallback retains the historical
unreported configured-max value of zero; selected_ubatch records the real size.

The consumer validates the envelope, required fields and successful layers,
rejects missing contracts or truncated files, and waits on incomplete final
lines. Unknown event names remain forward-compatible. The live boundary exposes
only groups, routes, layer costs and prefill summaries. It does not claim that
all historical log fields are a public API.

Tests check the reader on synthetic event streams, including reused token
positions, incomplete final lines, truncation and malformed events. A native unit
verifies production prefill events with debug channels off. Device readiness
additionally consumes a real short Linux model request's events.
