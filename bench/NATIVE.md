# Native final-benchmark wrapper

`native_wrapper.py --profile /absolute/path/profile.json` is an argv for
`run_final.py`, not a separate benchmark. It starts one owned server, checks its
health and listener ownership, invokes the existing `final_chat.py` exactly once,
and writes `wrapper.json` after cleanup. The driver supplies the `INFERSSD_*`
environment documented in [FINAL.md](FINAL.md), including its already-held lock
descriptor. The wrapper does not acquire a different lock, tune settings, build,
download models, purge page cache, pause applications or change power settings.

Prepare the profile outside the formal attempt. A profile is a reviewed launch
plan, not evidence that arbitrary values named “effective” are true. Use a known
build receipt and the production launcher's reviewed argv/environment. Capture
the launcher's configuration before the attempt; do not nest its `systemd-run`
inside this wrapper. Configure fixed budgets and context once for both tiers.
The full profile hash is part of their compared identity, so changing an argv,
environment value, file receipt or monitoring threshold invalidates the pair.

The public wrapper contains no machine/model presets. Paths and file hashes in
profiles and result artifacts are local operational data; review them before
publishing. Use a fresh output directory for every attempt and preserve failures.

## Profile

The following is a shape example, not a runnable configuration. It shows the
legacy environment-controlled form; current engine builds refuse retired
`GGML_MOE_*` variables, so use the
[typed extension](#typed-profile-extension-for-inferssd-ci) for them. Replace
every placeholder and enumerate **all** project shared libraries and model shards.
Byte budgets must come from the reviewed configuration. Never use zero to hide
an automatically sized cache: document its configured ceiling/reserve, and keep
its runtime capacity/admission telemetry in the server log. Zero means that a
tier is actually unused. The wrapper verifies the declared launch environment,
file identities and native resource envelope; it does not reconstruct the
engine's placement algorithm or prove every engine setting from telemetry.

```json
{
  "schema_version": 1,
  "platform": "linux",
  "revision": "FULL_40_CHARACTER_FINAL_SOURCE_REVISION",
  "binary": {"path": "/path/to/build/bin/llama-server", "sha256": "VERIFIED_SHA256"},
  "images": [
    {"path": "/path/to/build/bin/libggml-base.so", "sha256": "VERIFIED_SHA256"}
  ],
  "shards": [
    {"path": "/path/to/models/example.gguf", "sha256": "PREVIOUSLY_VERIFIED_SHA256",
     "dev": 1, "ino": 1, "bytes": 123, "mtime_ns": 123, "ctime_ns": 123}
  ],
  "template": {"path": "/path/to/template.jinja", "sha256": "VERIFIED_SHA256"},
  "alias": {"path": "/path/to/experts.tsv", "sha256": "VERIFIED_SHA256"},
  "args": ["-t", "2", "-tb", "2", "-b", "512", "-ub", "512"],
  "environment": {"GGML_MOE_EXPERT_GGUF_ALIAS_TSV": "/path/to/experts.tsv"},
  "resources": {
    "memory_budget_bytes": 8589934592,
    "host_reserve_bytes": 3221225472,
    "minimum_cpus": 2,
    "minimum_memlock_bytes": 268435456
  },
  "effective_settings": {
    "context_tokens": 8192,
    "source_mode": "direct-gguf",
    "h2d_limit_bytes_per_second": null,
    "expert_cache_budget_bytes": 1073741824,
    "vram_budget_bytes": 0,
    "unified_memory_budget_bytes": 0,
    "gpu_reserve_bytes": 0
  },
  "load_timeout_s": 600,
  "cleanup_timeout_s": 30
}
```

For a split GGUF, list shards in order from `00001-of-NNNNN`, with their original
verified device/inode/size/mtime/ctime and full SHA-256. The wrapper rechecks those
stat identities; it never hashes model weights during either arm. Keep model
files immutable. Hashes are trusted from the earlier complete verification;
stat continuity is not a defense against a privileged concurrent writer.
Small templates/alias indexes and build images are hashed before the request and
again after it. Alias rows must refer only to the pinned GGUF shards and remain
inside each source file. Linux may use `aliases: [file_spec, ...]` instead of
`alias`; the source environment must contain that exact ordered colon-separated
list. This supports separate expert and dense-source indexes. Mac uses one alias.

Do not put model/context/template/slot/host/port/cache/warmup/log-suppression
options in `args`; the wrapper owns them. It adds the first shard, explicit chat
template, `--reasoning off` (thinking off where the template has a switch), one
slot, fixed context, fit off, no warmup completion, no prompt cache and a private
loopback port. Library search is fixed to the binary's bundle. Preloads must be
individually pinned images; H2D governors are rejected. Ambient engine, BLAS,
OpenMP, CUDA-selection and loader overrides are not inherited: put intended values
explicitly in the reviewed environment.

## Platform ownership and resources

Linux requires a visible cgroup v2 root and delegated systemd user manager.
Each tier gets a unique scope with the declared hard `memory.max` and zero swap.
The existing lock open-file-description is transferred using `SCM_RIGHTS` to the
worker, then inherited by server/client. No second lock file is opened to bypass
an owner. The worker checks effective ancestor limits, OOM counters, CPU affinity,
host reserve, server scope membership and actual executable/project mappings.
Linux profiles must explicitly declare `minimum_memlock_bytes`, at least the
reviewed total staging/pinning requirement. Zero is valid only for a reviewed
configuration that needs no pinned memory. The worker records its soft/hard
`RLIMIT_MEMLOCK`, rejects an insufficient soft limit, and checks the server's
`/proc/PID/limits` inherited values. It never raises a limit. Arrange the required
limit before calling `run_final.py` (for example with an authorized `prlimit`
invocation); a systemd scope does not support setting `LimitMEMLOCK` on processes
that already exist. An unlimited value is recorded explicitly, not as zero.
Mappings must match every declared path, file device/inode and hash, including
the portable store or private liburing when present. CUDA driver/system images
are recorded in raw maps but are outside the project-image hash allowlist.

The controller stops and inspects only its unique unit. It retains its lease
while scope state is uncertain. A worker that loses its controller stops its
owned groups and retains its lease until no other scope process remains. This
is the shared ownership protocol described below, parameterized for the
profile's budget. It is not recovery after machine loss or arbitrary
noncooperating processes. A server that exits leaving descendants,
requires forced kill, changes identity or fails to exit normally cannot pass.

Mac profiles set `platform: "macos"` and add
`max_compressor_growth_bytes` and `max_swap_growth_bytes` under `resources`.
The memory budget is a **sampled soft RSS threshold**, not a cgroup or unified
memory allocation limit. The wrapper checks free+inactive pages as a reclaimable
reserve estimate, system compressor/swap growth and process RSS approximately
once per second. It preserves raw `vm_stat`/swap output and sample intervals;
samples are non-atomic and may miss short peaks. These counters overlap and must
not be summed. Use separately recorded engine cache/Metal settings to explain
unified-memory budgets. The OS can allocate outside the process RSS threshold.

Mac uses the existing `macos/measurement_images.py` actual `lsof` text-image
inspection, before and after the request, against the pinned CPU/Metal bundle.
It checks server PID/start identity, listener ownership and the inherited lock
descriptor. Its server/client groups are reaped before cleanup is confirmed;
no external native scope is created. A separate worker receives the same lock
through an authenticated Unix socket (kernel peer PID/UID and `SCM_RIGHTS`). If
the controller is killed, socket disconnect makes the worker clean its owned
groups while retaining the lock. If the worker dies without a cleanup receipt,
the controller retains its lock in quarantine. Killing both guardians or a
machine failure requires manual recovery; no such state is called a pass.
It does not auto-pause background software.

Both adapters retain server/client logs, launch inputs, source identities, native
resource observations and cleanup evidence. The production server log must show
zero completion tasks before the formal client and exactly one afterwards.
Initialization reads are allowed; a fresh process does not imply an empty OS
page cache. Receipt validation cannot turn a failed measurement into a pass.

## Validation boundary

`bench/tests/test_native_*.py` covers fixed profile/source identities, real tiny
child ownership and lock inheritance, controller SIGKILL cleanup, unknown-worker
quarantine, mock two-tier HTTP results through the
unchanged formal client, partial/timeout/interrupted requests, leftover children,
foreign/missing/changed image maps, cgroup/ancestor faults and Mac soft thresholds.
Native Mac PID/listener/FD tests run only on Darwin and report a skip elsewhere.
Linux scope logic has mock tests; an actual harmless scope/lease lifecycle check
is still required on each intended Linux host before the final attempt. These
tests do not load models or constitute final benchmark data.

## Shared ownership code and deployment

The native wrapper uses the same `inferssd.platform.lease` and
`inferssd.platform.process` and `inferssd.platform.guardian` implementations as the framework. With a normal wheel
installation, the wrapper imports those modules and records their actual file
SHA-256 identities in `native_ownership_core`. Its required ownership API is 1;
an absent package or unsupported API fails before launch.

For standalone deployment, prepare a new bundle explicitly:

```sh
python3 bench/bundle_native.py --output /path/to/new-wrapper-bundle \
  --core /path/to/reviewed/inferssd/platform
```

Use the exact `inferssd/platform` directory from the reviewed source or installed
wheel. Copy the **entire** resulting directory, including `ownership/`,
`native-core.json` and `bundle-manifest.json`, and verify every manifest member
before an attempt. The wrapper loads that adjacent manifest automatically;
`INFERSSD_NATIVE_CORE_MANIFEST` can select an explicit manifest instead. Every
member is verified before either module is imported. A missing, changed or
unsupported bundle fails; it never falls back to another installed copy. No
search-path modification or ambient checkout is used. Archive the complete
bundle and manifest with the result. Timing, request and prompt protocols are
unchanged.

## Typed profile extension for InferSSD CI

A new production profile can add `typed_settings` as an absolute `{path,sha256}`
file specification. Its contents use the current `inferssd-settings 1` parser
and complete registry. `expert.source` must equal the exact ordered
`alias`/`aliases` paths;
retired control environments are refused. The wrapper owns `--ssd-settings` and
`--ssd-telemetry`, creates a fresh native-events path per arm, and checks the
actual native `ssd.settings` publication against the pinned typed values. The
settings file identity is checked before/after the request with the other small
assets. The receipt includes typed values and exact launch policy in addition
to the existing actual resource/build/model evidence. Legacy profiles remain a
separate historical wrapper contract; InferSSD CI requires the typed extension.
