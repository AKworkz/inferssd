# Original launcher observations

Expected outputs were produced only by the recorded reference revision `8b12192920b339a4ef31c9006d1ebe20304ce45a`, before the typed resolver existed. `oracle/launch-manifest.json` binds every original source. Linux uses Bash 5.2.21 on a Linux fixture host; Mac observations execute the original Python module with explicit Darwin/arm64 facts and intercepted process effects, under Python 3.9 and 3.12. No model or native engine is run.

The 58 Linux cases include repeated index reuse/invalidation and preserve complete stdout/stderr/status, metadata command replies, ordered stub calls and artifact bytes. Native/service commands are fail-closed sentinels. CPU/memory/CUDA/port/service replies are fixture facts, not measurements of a GPU. The Linux uid and cgroup-v2 existence are explicit observations, not invented Mac behavior. The 28 Darwin cases capture would-exec argv/env, filesystem publication and all defined errors; they do not claim Apple hardware execution.

Only fixture-root and interpreter spelling are replaced with named placeholders. Mac's random temporary directory and path-derived alias digest are canonicalized explicitly; neither is an algorithm output under comparison. Fixed-path tests separately check the digest and publication paths. Numeric results, argument order, receipts, diagnostics and partial artifacts are not normalized. The reference's die helper includes the exit-code argument in some messages; those trailing `2` strings are retained.

The Linux expected stderr leaves out two of the reference's messages, on purpose. The engine's CUDA backend now reads experts through the SSD executor's io_uring ring, without liburing, so the port no longer greps `libggml-cuda.so` for `io_uring_queue_init`: it neither warns that the GPU prefill route reads with synchronous preads (7 Linux cases and both composed-settings runs) nor refuses DENSE_HOMES weights on the drive (`dense-no-uring`). That case's output was regenerated from the port and differs from the reference's only by the refusal line. The recorded stub calls keep the reference's grep, which the order test skips.

The original shell suites contain 102 settings assertions plus 6 replica cases (108 total). The original suite source is preserved unchanged. Its replica rules are not used to generate the new resolver's expected output.

The reference source closure includes `settings.py` and its imports, which four settings-generation assertions require. No production source was changed to make a test pass.
