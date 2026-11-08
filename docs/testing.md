# Test gates

The `unit` CTest label contains only model-free tests that require neither a GPU
nor cgroup setup. Build the targets before running `ctest -L unit`; a missing
executable, skipped test or zero selected tests is not a passing gate. CI uses an
explicit subset to keep the default check small. The CUDA workflow compiles the
server for `sm_120a` and `sm_121a` on `main` and on pull requests that change the
engine or build configuration; it does not establish GPU runtime correctness.
Device runs must record both the selected and actually executed tests.

The CPU sink suite runs on Linux, macOS and Windows against the same ggml
reference graphs, bit for bit. It exercises direct reads, the OS read ring
(io_uring on Linux, IoRing on Windows), affinity, the host memory envelope and
experimental scheduling paths; on Linux also cgroup memory limits. It is not
interchangeable with the portable unit gate. Do not interpret a full-suite run on a small hosted runner as
a production device acceptance test. GPU and cgroup suites need dedicated launchers
and explicit resource prerequisites before they can become automated gates.

Some bundled upstream tests in a full run depend on outside resources: `test-download-model`
fetches the model that `test-thread-safety` and `test-state-restore-fragmented` use,
`test-tokenizers-ggml-vocabs` clones vocabularies with Git LFS, `test-quant-type-selection`
compares its recorded snapshots with the current Hugging Face files, and `test-jinja-py`
needs `python3` with `jinja2`; exclude them with `ctest -E` where these are unavailable or have changed.

## Shared expert store

`LLAMA_SSD_STORE` is on by default on Linux, macOS and Windows. The store is a
static library that only llama links; each context owns its source index. The
supported configuration uses bundled ggml, shared or static libraries, the CPU
backend, and linked backends (`GGML_BACKEND_DL=OFF`, `GGML_CPU_ALL_VARIANTS=OFF`).
The CUDA SSD transport (`GGML_CUDA_KIMI_VRAM_TIER=ON`, set by the `linux-cuda`
preset) reads through the same read ring as the CPU sink, so no build needs
liburing: on Linux the ring submits io_uring requests through system calls. System-ggml and runtime backend-plugin variants remain
outside this store's supported build boundary. Every OS runs configured experts
through the store and the CPU sink. The OFF configuration is retained for explicit legacy
comparison, not as a runtime fallback. `ssd_source_index` checks the store end to
end on every OS.

The legacy store fixtures below are Linux-only. `test-c0-store` and `test-c0-config` link the actual production store target and
compare it with a fixed test reference. `test-c0-parser-failures` injects
allocation and callback failures into the parser to verify handle cleanup.
Their CTest names `c0-store`, `c0-config` and `c0-parser-failures` carry `unit`
and run in hosted CPU CI. The cross-DSO/mock-provider, source-open-hook and direct
I/O fixtures carry `device`; they do not silently join the hosted unit gate.
The production store CMake target does not read the fixture's reference includes
or execute Python. The fixed reference is copied only in the test configuration.

Successful shared initialization emits one receipt:
`[ssd_store] provider=cuda|portable sources=N entries=N published=1`.
Together with loaded-library hashes and direct-file descriptor identity, this
records which provider populated the single owner. It is not a replacement for
the finite-RAM same-sink numerical comparison or normal reader-shutdown checks.

## Device failures

Device failures are not allowlisted. Record each one with its full test name,
reproduction conditions and outcome on the current hardware. A test with a
hardware-specific numeric expectation must document that expectation rather
than become a generic skip.

## Resource prerequisites

The registered sink/prefill I/O fixtures carry `device`; cache-admission cases
additionally carry `cgroup` on Linux and `envelope` on macOS and Windows, where
the native host ledger always has one. The metadata-only prefill graph and cache
compute-policy tests carry `unit`. Modes that need a Linux facility (cgroup v2
limits, `/proc`, forked `RLIMIT_MEMLOCK` children, pread interposition) return
77 elsewhere; a registered case with such a part prints `status=SKIP` for that
part and runs the rest. Existing manual CUDA workloads remain manual; adding
labels does not authorize a device run.

A fixture returns 77 with a reason for insufficient CPU affinity (compute and I/O
participants are reported separately), unavailable io_uring setup, unsupported
aligned direct reads on its actual input filesystem, or a missing required finite
cgroup on Linux. An unreadable native envelope on macOS or Windows is a failure.
The finite-cgroup probe does not infer a payload budget: fixtures may change
reserves or intentionally exercise pressure. Their explicit size and owned-scope
rules remain authoritative, including the load-room minimum check. Use the
intended storage directory through `TMPDIR` (`TMP` on Windows); probing another
mount does not establish that this fixture's filesystem supports direct I/O.

Fixed-buffer tests use the actual `IORING_REGISTER_BUFFERS` result, including the
real ring geometry and concurrently live rings. EPERM or ENOMEM there records a
skip and the current `RLIMIT_MEMLOCK`; a tiny successful registration is not
proof of the full pin budget. Negative memlock tests still deliberately lower
the child limit and assert rejection. Malformed settings, missing files, short
reads, ordinary allocation errors, registration EINVAL/EFAULT, and numerical
mismatches remain failures. No runtime fallback is added.

CTest preserves 77 as skipped in JUnit. The device gate compares every status with
an explicit reviewed baseline and rejects an all-skipped selection. ASan cannot
establish the prefill cache's process-RSS release assertion: that lifecycle test
executes its other assertions but returns 77, requiring a separate normal-build
pass for physical-release acceptance. Device allowlists and hosted runner
capabilities must be measured on the current hardware, not inferred from
historical failure totals; none is recorded in this repository yet.

## Apple Silicon integration

macOS runs configured experts on the same CPU sink and store as Linux and
Windows, so the sink, prefill, store, read-ring and platform suites above are its
expert-path tests. `test-llama-mmap-dontneed` separately verifies Apple's mapping
advice with a tiny temporary file; production memory/performance effects still
require model runs. Launcher and metadata validation tests run with
`python3 -m pytest -q tests/integration/test-macos-launcher.py`.

The Mac CI build enables Metal and runs a hosted-runner subset of those suites:
the sink at one to three ggml threads, its SITU, SWIGLU_CLAMP and SWIGLU activations,
Muse Spark's experts with their post norm and grouped prefill, bit for bit against
ggml reference graphs, the prefill llama graph, and the store's source index, read ring and platform layer. It also
exercises Muse rendering with `test-chat --template muse-spark` and checks the
Q4_K/Q5_K/Q8_0 CPU `MUL_MAT_ID` cases. This is a compilation and synthetic
correctness gate. The separate target-device gate still needs matched model token
sequences and memory/performance measurements; no model result follows from
hosted CI.

The unfiltered `test-chat` suite is outside the supported Muse rendering gate.
It is not declared passing or silently allowlisted.

## Windows

The Windows CI job runs the framework tests, then builds the engine as
`install.ps1` does (LLVM clang with the Visual Studio C++ tools' environment,
without a GPU backend) and runs the synthetic and platform tests of the Linux
job that Windows builds, the sink and grouped prefill tests of the Mac job, and
the Windows host budget, platform file and both read ring backends. A test that
builds llama's internal classes stays off Windows, because `llama.dll` does not
export them. Like the other hosted jobs, it is a compilation and synthetic
correctness gate.

The CUDA workflow also builds `windows-cuda` (LLVM clang for the C++, `nvcc` with
the Visual Studio host compiler) from NVIDIA's redistributable toolkit archives.
It compiles and links only; no GPU runs it.
