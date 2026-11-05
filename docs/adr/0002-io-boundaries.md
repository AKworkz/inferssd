# 0002: Preserve the two Linux read engines

Status: superseded by [0005](0005-one-read-ring.md).

CPU direct-syscall io_uring and GPU liburing keep their existing scheduling and
completion ordering in separate core I/O modules. CUDA owns pinned stages and
events; the core owns no CUDA resources. Source descriptors are borrowed from an
immutable context index. Configured read failures propagate; they do not silently
switch to mapped payloads. Unread completions and seed consumers must drain
before storage is reused or released. Mac uses its pread/F_NOCACHE source.
