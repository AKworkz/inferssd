# Mac expert-source experiments

`measure.py` is a legacy experiment driver for the former Mac expert source, an
LFU cache behind a CPU `MUL_MAT_ID` hook. It configures that source through
`SSD_LLAMA_MAC_*` and `GGML_MOE_*` environment variables, which current engine
builds refuse at startup as retired controls. It applies only to builds that
predate typed settings and still contain that source. macOS now runs configured
experts on the same CPU sink as Linux and Windows; current launches use the
settings described in [configuration](../../docs/configuration.md) and report
through native telemetry rather than the `[mac_expert_pack]` counters below.

`measure.py` runs one fresh server for a specified model and configuration.
The first request uses a fresh server, including its normal initialization probes;
later requests reuse its application cache but disable prompt/KV reuse. Results label
these requests `fresh-server` and `same-server-repeat`, respectively. A fresh
server does not imply a completely untouched expert cache. No OS page-cache purge is performed.
Record the physical disk counters rather than calling the first request a cold SSD test.

The experimental `SSD_LLAMA_MAC_EXPERT_READ_AHEAD` setting accepts integers from 0 through 4.
Zero retains synchronous reads. Nonzero values bound the number of outstanding
reads for experts already selected by the current operator. Staging is additionally
limited to 64 MiB per operator. Expert arithmetic, execution order, cache heat and
admission order remain unchanged. This setting exists for controlled experiments;
a production default requires numerical, memory and end-to-end validation.

`GGML_MOE_EXPERT_GGUF_ALIAS_TSV` selects the same index format as the Linux reader.
Generate the index from GGUF metadata with
`python3 -m inferssd.source.gguf_alias --model-glob 'MODEL-*.gguf' --out alias.tsv --summary-json alias.json`,
the generator a launch runs when `expert.source` is not set.
It is mutually exclusive with `SSD_LLAMA_MAC_EXPERT_PACK`. GGUF slices can be unaligned
and end exactly at EOF. With read alignment 1, the reader requests logical bytes.
Alignment 4096 or 16384 expands the source interval to page boundaries, capped at
the source file's EOF; the caller still receives exactly the logical slice. Cache
accounting includes the whole allocation. The 64 MiB staging bound applies per
operator batch; callers running multiple contexts must include each batch in their
memory budget.

Counters are cumulative. `read_call_ms` sums individual pread service times across
workers and is not elapsed I/O time or a device bandwidth denominator. `read_wait_ms`
measures demand-side blocking; `compute_ms` measures the ggml lead thread's work and
`barrier_ms` its synchronization waits, including waits for other computing threads.
`acquire_ms` includes cache policy, staging setup and read waits. Lookup, policy and
allocation counters are nested diagnostics and must not all be added as disjoint
request phases. Physical bytes divided by the corresponding phase wall time give
observed device throughput estimates. System snapshots are non-atomic: each records
`sample_begin` and `sample_end` around the IOKit, VM and process queries. The first
token requests an asynchronous snapshot so it does not block the SSE reader;
`first_sample_delay_s` exposes the sampling interval relative to that token. Use
these bounds when assigning physical bytes to phases. Memory sampling records RSS,
compression and swap; one-second sampling can miss shorter peaks.

With `--trace`, four files are written on normal shutdown: the legacy request list,
`.requests.tsv` with demand timestamps and hit flags, `.reads.tsv` with actual
source extents, read timestamps and call durations, and `.phases.tsv` with cumulative
lead-thread phase counters. These optional traces accumulate in memory until
shutdown, so use short requests when collecting them. They allow per-request
logical miss-byte accounting and bounded replay of real misses without duplicating weights.
For a streamed request of N output tokens, the first token follows prefill and only
N-1 subsequent decode evaluations run; use that count for per-decode-step bytes.
The server's raw `timings` uses N in its reported generation rate. The separate
`client_decode_tokens_per_second` uses N-1 divided by the interval from the first
to the last token event; completion/stop latency remains in request `wall_s`.
Neither interval makes asynchronous device snapshots exact token boundaries.

`--binary` selects a build bundle. The driver clears inherited `DYLD_*` overrides,
then sets `DYLD_LIBRARY_PATH` to its adjacent libraries. This expresses a selection
intention. Outside timed requests, `lsof` snapshots verify the actual executable
and project libraries against that bundle's device/inode and file SHA256 identities.
Core libraries are required after health; CPU/Metal backends are required after
requests, allowing late loading. Previously observed images must remain unchanged.
Raw lsof output and failure receipts are retained. These are point-in-time,
file-backed checks, not in-memory text hashes or a guarantee against later loading;
the system dyld shared cache is outside their scope. Keep build files unchanged and
preserve separate build directories for baseline and candidate binaries. The report
also records alias/prompt hashes, driver source revision and uncommitted diff.
The driver supplies the selected model's chat template, disables reasoning, and
explicitly places DeepSeek Engram embeddings on CPU, matching the normal launcher. `--poll` overrides both decode and batch threadpool polling;
omitting it retains the server default. Use the same prompt, output length and
parameters for each comparison. One-second memory samples within recorded requests
exclude loading and later `vmmap` inspection, and can miss shorter peaks.

`--ctx-checkpoints` optionally overrides the server checkpoint count; omitting it
preserves the server default. Across engine revisions, checkpoint support can
change the actual prefill batch boundaries even with the same prompt and `--batch`.
For a numerical comparison, verify those boundaries; `--ctx-checkpoints 0` on both
sides can remove checkpoint-induced splits. Keep this controlled comparison
separate from default-configuration validation, and record the effective arguments.
Different batch shapes can select different floating-point operation orders.

`--teacher-reference /path/to/results.json --runs 1` evaluates fixed continuation
prefixes from a prior run and records conditional raw top-5 log probabilities.
It binds the reference results and prompt file by SHA256 and validates each request
independently. `post_sampling_probs=false` is explicit for all requests. Its request
pattern differs from ordinary generation, so its timings are not a performance
baseline. `--profile-seconds 3` captures separate native prefill and decode stack
samples; profiling runs are likewise diagnostic, not throughput results.

`--pause-pid` only accepts this user's screenpipe process. If it is running, the
driver records its PID, UID, start identity and prior stopped state, temporarily
stops it during measurement, and resumes the same process during normal or handled
termination. A previously stopped process is left stopped. Python signal exceptions are
deferred while publishing pause/child-process responsibility and while independently cleaning up
the screen recorder, monitor, profiles and server. Children inherit unblocked signal masks. Cleanup failures are recorded;
they cannot turn a partially completed run into a successful receipt. SIGKILL,
host shutdown or a failure to verify process identity cannot guarantee restoration.
Global device and memory counters can include unrelated system activity; record
that limitation when interpreting them.

Only `receipt.json` with `status=complete` confirms protocol and evidence completion.
Every HTTP request must return the expected token count, one validated five-entry
`top_logprobs` record per token, matching token IDs, an untruncated limit stop, and
valid timing counts/values. A total request deadline supplements urllib's idle
socket timeout. Server shutdown must finish normally, requested profiles must
succeed, and alias trace row/byte counts must match shutdown counters. Partial
responses and errors remain available for diagnosis. A complete receipt does not
assert numerical equivalence: compare tokens and raw log probabilities with a
matching baseline separately. Teacher/profile receipts are explicitly ineligible
for throughput comparisons.

Run protocol and image-identity tests without model weights:

```sh
python3 -m pytest -q tests/integration/test-macos-measurement.py \
  tests/integration/test-macos-measurement-images.py
```
