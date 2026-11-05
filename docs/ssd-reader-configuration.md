# Typed reader configuration

The CPU reader accepts an owned, immutable settings snapshot through
`ssd::cpu::create_configured`. It copies settings and debug channels before
starting workers. The context owns the telemetry writer and resolver; it must
destroy the reader, including its I/O, prefetch and compute workers, before
destroying either borrowed service. `runtime-services.h` documents that ordering.

The supported execution profile uses eight ring experts, the ordered split-CV
scheduler, early up/gate, role-major and nonblocking down scheduling, dynamic
rows in chunks of 64, count-decay RAM replacement, uniform allocation, retention
across requests, partial reclaim and reclaimable file accounting. The cgroup
reserve is 512 MiB. Typed CPU affinity, participant count, prefetch depth,
placement costs, host headroom, GPU budget, slot cap and margins remain inputs.
Hardware topology, memory limits and device sharing are measured from the real
host. Readers do not read environment variables.

The null-reader admission callbacks take the same context snapshot. They build
an independent ledger with the explicit host headroom and device reserve, so an
early graph allocation cannot silently use another context's defaults. Mutable
prefill pages and interval accounting remain local; process-wide admission
belongs to the separate host arbiter.

The CUDA reader uses a same-build typed-settings extension and refuses an absent
or incompatible extension. Its supported profile fixes asynchronous/deferred
uploads, retained elastic global residency, count decay and measured admission.
Debug observation is copied separately.
The original untyped GPU creation ABI slot returns an explicit error. It remains
non-null so older ABI discovery cannot dereference a missing function slot.

The CPU reader emits versioned `ssd.io.read` and `ssd.place.layer` events after
completion and drain, including request/completion/pending counts, bytes, cache
results, wall/wait/compute time and thread CPU time. A writer failure is returned
to the caller. The context integration emits the initial effective settings and
their sources; this reader module does not create another telemetry owner.

## Fixture construction

Private `ssd/testing.h` and `ggml-ssd/testing/tier.h` structures preserve explicit
inputs for numerical, scheduling and lifecycle fixtures. They construct the real
reader and real CUDA format engines. Product factories derive their canonical
policy directly from typed snapshots and do not consume these fixture objects.

`fixture_inputs` is owned by each test invocation and passed through constructors
and calls. CPU CTest matrix variants use `--fixture key=value`; unknown fields
are rejected. The CTest matrix retains transport, split-CV, scheduler, replacement
and allocation dimensions without process environment mutation. The retained
cache-compute algorithm tests pass explicit algorithm objects; supported product
configuration leaves that experiment off.

The prefetch residency unit keeps its existing stand-in residency table and real
CPU file reads. It is a decision-policy unit, not CUDA numerical or performance
evidence. Real CUDA numerical and failure tests continue to use the production
DSOs and require an explicitly owned device.

The fixed-buffer failure child creates its real io_uring first, then lowers only
the child's memlock limit and capability. Some kernels charge ring creation to
memlock; applying the fault later reaches the intended buffer-registration
failure while preserving direct cache reads and fail-closed ring bypass checks.

## Integration and validation boundary

The reader depends on the typed settings and telemetry modules.
CLI/context/provider wiring, Mac source configuration, process-wide memory
admission and the Python launcher are separate modules. Test-only policy inputs are not an
environment translation layer.

Linux validation uses real io_uring and fresh 4 GiB memory scopes with swap
disabled. The configured-reader gate compares two contexts against the CPU
reference, changes the caller's settings/debug objects after construction,
checks pre-worker affinity refusal, early admission, real telemetry write
failure, complete drain and joined workers. Additional gates cover direct and
fixed reads, cache leases/reclaim, grouped prefill, trace/profile and prefetch
fault paths. CUDA host syntax uses only official SDK headers; it does not count
as CUDA execution. Accepting a CUDA build also requires the numerical and
lifecycle checks on an owned device. These are development checks and supply no
formal benchmark samples.
