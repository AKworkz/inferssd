# Final performance benchmark

`run_final.py` runs one device/model pair on the final source: the immutable
`prompt-512.txt` and `prompt-4096.txt` files, in that order, once each. Each produces
exactly 256 tokens. Repeat the invocation for each supported device/model pair;
this tool does not discover devices, tune settings, retry, average or select runs.
No formal performance data is included with the driver.

The labels describe approximate input sizes. `final_chat.py` asks the model's
`/apply-template` endpoint for authoritative IDs with one user message and the least
thinking the chat template offers, as `/props` reports it: thinking off, or the first
`reasoning_effort` of a template that always thinks, as Muse's does. It sends those IDs
unchanged to native `/completion`. `/props` must
report exactly one slot with space for **actual input count + 256**. An actual
4K-tier prompt may exceed 4096 tokens; configure context accordingly beforehand.
The bundled prompt SHA-256 values are enforced before any completion request.

Sampling is greedy top-k one, seed zero, EOS ignored and prompt caching disabled.
The client requires every token, its pre-sampling top-five record, a final length
stop, no truncation, matching counts, and finite positive server timing. It saves
original SSE bytes, timestamped events, exact IDs, protocol and failed results.
`stream_chat.py` and `run_abba.py` still reject inputs over 1024 tokens; there is no
new long-input option on either diagnostic entry point.

## Run contract

Prepare and review a device-specific wrapper first. It must start a fresh owned
server, load the intended model/template with fixed budgets/settings, wait for
health, run **one completion only**, and stop/reap its owned processes before
writing its receipt. Initialization reads are allowed; warm-up completions are
not. Each tier invokes the same wrapper argv with a new artifact directory.
The shared [native wrapper](NATIVE.md) provides Linux scope and Mac process
adapters with an explicit, reviewed profile; it has no device presets.
Use the intended current GGUF shards directly, with no legacy expert pack and no
artificial H2D transfer cap. Keep host RAM budgets matched for any separately
declared comparison and retain effective cache, VRAM and unified-memory budgets.
The wrapper must not start another benchmark, hash model weights, or compile
during the timed request. It can reference previously verified model receipts
after confirming the files still match their recorded identity.

The wrapper runs the client as follows after its own health check:

```sh
python3 bench/final_chat.py --url "$SERVER_URL" \
  --tier "$INFERSSD_TIER" --output "$INFERSSD_RESULT_PATH" \
  --timeout "$INFERSSD_REQUEST_TIMEOUT"
```

It must capture the client's return code and perform cleanup even on failure.
Do not accept an already listening port as evidence of server ownership.
The wrapper remains responsible for native resource limits, memory reserves and
platform-specific process scopes. Its assertions and raw evidence are reviewed
host-side facts; the generic driver does not attest hardware or recreate those
checks merely by validating JSON.

An example configuration (replace labels, revision and wrapper path) is:

```json
{
  "device": "device-label",
  "model": "model-label",
  "revision": "FULL_40_CHARACTER_FINAL_SOURCE_REVISION",
  "argv": ["bash", "path/to/reviewed-device-wrapper.sh"],
  "request_timeout_s": 1800,
  "arm_timeout_s": 3600
}
```

Both deadlines are finite and limited to 1--7200 seconds. The arm deadline must
exceed the request deadline, covering load, health, request and normal cleanup;
bounded emergency cleanup follows on timeout. The request deadline covers
templating, properties and the complete SSE operation, not just socket idleness.
Choose deadlines before the attempt; the defaults allow slower models more than
the short driver's 300 seconds. Do not extrapolate partial output after timeout.

```sh
python3 bench/run_final.py --config final-device-model.json \
  --output bench/runs/final-device-model-attempt-001 \
  --lock /path/to/existing-device.lock --stop /path/to/STOP
```

The output directory must be new. The driver holds a nonblocking exclusive flock
on the existing inode across both tiers, without truncating or unlinking it.
It checks STOP before and during each arm and handles INT, TERM and HUP as stop
intent. Any failure stops the sequence and records the other tier as `not_run`.
Failed attempts must remain separate if a later fix justifies another attempt.
No attempt here constitutes a numerical equivalence test or a CTest device gate.

The wrapper receives `INFERSSD_RUN_ID`, `INFERSSD_RUN_DIR`, `INFERSSD_RESULT_PATH`,
`INFERSSD_WRAPPER_RECEIPT`, `INFERSSD_TIER`, `INFERSSD_PROMPT_PATH`,
`INFERSSD_REQUEST_TIMEOUT`, `INFERSSD_OUTPUT_TOKENS=256`, `INFERSSD_DEVICE`,
`INFERSSD_MODEL`, `INFERSSD_REVISION` and `INFERSSD_LOCK_FD`. Paths are absolute except
the operator's argv, which retains the invocation working directory. The lock FD
is inherited by the wrapper; preserve that same open-file description in owned
servers/lease holders where the device launcher supports it. Do not open a
replacement lock to bypass an owner. A systemd scope must retain the reviewed
scope/lease cleanup protocol: process-group cleanup alone cannot find a server
that moved out of the group.

The generic driver terminates only its newly created wrapper group. A wrapper
that leaves descendants, lacks a cleanup receipt or cannot prove scope cleanup
fails acceptance. Unconfirmed cleanup also creates STOP to quarantine the device;
inspect the exact owned scope/processes before clearing it. This is not a claim
of recovery after machine loss, SIGKILL, or a noncooperating external launcher.

## Wrapper receipt

Write `INFERSSD_WRAPPER_RECEIPT` after cleanup. Every following field is required
on success; hashes must be lowercase SHA-256. `evidence` paths are relative to
the arm directory and must exist with matching hashes. Preserve the raw commands,
server logs, actual library inspection, model identity and resource observations.
Do not substitute adjacent-library hashes for actual loaded-library identity.

```json
{
  "schema_version": 1,
  "status": "passed",
  "protocol": "final-256-v1",
  "tier": "512",
  "device": "device-label",
  "model": "model-label",
  "revision": "FULL_40_CHARACTER_FINAL_SOURCE_REVISION",
  "server": {
    "pid": 12345,
    "start_identity": "OS process start identity",
    "argv": ["path/to/llama-server", "actual", "arguments"],
    "fresh_process": true,
    "completion_requests": 1,
    "returncode": 0,
    "owned_processes_gone": true,
    "resource_scope_gone": true
  },
  "build": {
    "revision": "FULL_40_CHARACTER_FINAL_SOURCE_REVISION",
    "binary_sha256": "REPLACE_WITH_HASH",
    "actual_loaded_libraries_verified": true,
    "loaded_libraries": [{"path": "actual/library/path", "sha256": "REPLACE_WITH_HASH"}]
  },
  "model_identity": {
    "verified": true,
    "shards": [{"path": "model-shard.gguf", "bytes": 123, "sha256": "REPLACE_WITH_HASH"}]
  },
  "host": {
    "hardware": "CPU/GPU/RAM/storage identity; details in evidence",
    "os": "OS/kernel/driver identity",
    "memory_budget_bytes": 123,
    "resource_checks_passed": true,
    "effective_settings": {
      "context_tokens": 8192,
      "source_mode": "direct-gguf",
      "h2d_limit_bytes_per_second": null,
      "expert_cache_budget_bytes": 0,
      "vram_budget_bytes": 0,
      "unified_memory_budget_bytes": 0,
      "gpu_reserve_bytes": 0,
      "other_effective_settings": "record here"
    },
    "os_page_cache_treatment": "none; OS cache state uncontrolled"
  },
  "evidence": [{"path": "host-receipt.json", "sha256": "REPLACE_WITH_HASH"}]
}
```

List all model shards and all loaded project libraries in stable order. Build,
model, host, effective settings, budgets and page-cache treatment must match
between tiers; PID/start identity must differ. The budget fields must be actual
nonnegative byte values; use zero only when a tier is unused/inapplicable, and
`null` for the required absence of an H2D cap. Set `resource_scope_gone` true
only after checking an external scope is empty, or after establishing that no
external scope was created. A failed measurement still writes a receipt with
its failure and the two cleanup booleans; missing cleanup evidence quarantines
the device. Clean shutdown does not turn a failed measurement into a pass.

## Results and interpretation

`manifest.json` retains the config. `attempts.jsonl` and `summary.json` retain both
ordered rows, including failure/not-run rows and receipt hashes. The driver
rebuilds each successful result from raw SSE and timestamped events, verifies
fixed request parameters and all counts, and validates the wrapper contract.
Each performance row has `n=1`; no variance or significance is reported.

For token arrival times `t1` through `t256`, `result.metrics` contains:

| Field | Definition |
| --- | --- |
| `server_prefill_tps` | actual input count / server prompt seconds |
| `ttft_s` | completion dispatch to first token |
| `decode_first128_tps` | 127 / (t128 - t1) |
| `decode_second128_tps` | 128 / (t256 - t128) |
| `decode_tps` | 255 / (t256 - t1) |
| `server_native_decode_tps` | server predicted count 256 / server predicted seconds |
| `end_to_end_output_tps` | 256 / complete request wall seconds |
| `request_wall_s` | completion dispatch through final stream termination |

The first window has 127 intervals; the second includes the interval across the
128/129 boundary. Final stop-event delay affects request wall time, not decode
intervals. Native server clocks and client arrival clocks remain distinct.
Client transport buffering/evidence recording can affect observed intervals.
A fresh process proves a new application lifetime, not an empty OS page cache
or an absence of initialization reads. State the actual cache treatment and
publish provenance/raw records only after removing private operational data.
