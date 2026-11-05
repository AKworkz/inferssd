# 0005: One read ring per OS for every reader

Status: accepted. Supersedes [0002](0002-io-boundaries.md).

Every reader of expert and dense bytes reads through the read ring of its OS:
io_uring through system calls on Linux, IoRing (or overlapped reads) on Windows,
reader threads on macOS. The CPU sink, the GPU lane and the CUDA transport each
own an instance and keep their own scheduling and completion order; their queues
are not merged. CUDA owns its pinned stages and events; the core owns no CUDA
resources. Source descriptors are borrowed from an immutable context index.
Configured read failures propagate; they do not silently switch to mapped
payloads. Unread completions and seed consumers must drain before storage is
reused or released.

The CUDA transport no longer has a liburing reader of its own. Its production
path issued plain reads, one batch per submit, which the Linux ring issues the
same way, so no build needs liburing.
