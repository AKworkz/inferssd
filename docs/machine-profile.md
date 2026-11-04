<!-- SPDX-License-Identifier: Apache-2.0 -->
# Machine profile

InferSSD measures each machine once and keeps the result in a characteristics
file. The file describes the machine only: its CPUs, memory, GPUs and
the drives that hold models. It holds no model data, no placement and no cache
sizes; each launch works those out from the model's GGUF header and this file.

You do not need to run anything for this: the first `inferssd serve` or `cli` on
a machine measures it, and later launches reuse the file (see
[launches](#launches)). `inferssd profile` measures or shows it on its own.

Linux and macOS are supported. Windows is not supported yet.

## Usage

```sh
# Measure the machine and the drive that holds a model.
inferssd profile --model /models/model-00001-of-00004.gguf --build build-linux-cuda

# Print the saved profile without measuring anything.
inferssd profile --show

# Measure the drive of a models directory, then print the profile and the
# supported models' speed estimates as one JSON object.
inferssd profile --model ~/models --json

# Measure every section again.
inferssd profile --reprofile --model /models/model-00001-of-00004.gguf
```

| Option | Meaning |
| --- | --- |
| `--model PATH` | A model file or directory whose drive is measured. Repeat it for models on other drives. A file named like `name-00001-of-00004.gguf` brings all shards of its set; a directory brings every file below it, and one without a file of 1 GiB is measured with a test file (see [drive measurement](#drive-measurement)). The [speed estimates](#speed-estimates) take the models to be on the drive that holds most of the given files, or with none, on the first directory's. |
| `--resources FILE` | An [engine resource descriptor](engine-resources.md) whose `probes` map names the probe executables. |
| `--build DIR` | An engine build directory; the probes are `DIR/bin/inferssd-*probe`. |
| `--reprofile` | Ignore the saved file and measure every section. |
| `--output PATH` | Use this file instead of the default location. |
| `--show` | Print the saved file's summary and exit; with `--model`, also the estimates for that drive. Measures nothing. |
| `--json` | Print the summary and the estimates as one JSON object on stdout instead of the text lines. |

Without `--resources` or `--build`, the command uses the installed engine, as
`cli` and `serve` do: `INFERSSD_RESOURCES` when nonempty (set by the one-command
installer's wrapper), otherwise `current/engine-resources.json` in the data
directory. Without an installed engine it exits with status 2. `--show` only
reads the saved profile and needs none of these.
The command prints one progress line per step on stderr and one summary line per
section on stdout, with `--model` one estimate line per supported model, then
names the sections of this run that are partial or failed (on stderr with
`--json`). It exits with status 1 when a section measured in this run failed
(the file is still saved, with the reason recorded), and with status 2 on a
usage error, a `--model` path that does not exist, or when another
`inferssd profile` is running.

## Launches

Before a launch starts the server, it checks that the
characteristics file exists and is current for this machine and for the drive
that holds the model. When it is not, the launch measures what is missing
first, printing its progress on stderr. On a new machine this takes one to
three minutes; later launches find the file current and measure nothing.

The launch finds the probes next to the server executable, then in the
[engine resource descriptor](engine-resources.md)'s `probes` map, then in the
checkout's build directories. A probe that is missing or fails is reported as
a warning: its section is recorded as failed and the launch continues without
what it would have measured.

| Option | Meaning |
| --- | --- |
| `--reprofile` | Measure every section again before launching. |
| `--machine PATH` | Use this characteristics file as it is and measure nothing, for example one file measured once and shared by several runs. A dry-run reads it too. |

Each launch then combines the file with the model's GGUF header and with what
it reads at that moment (the CPUs it may use, the usable GPUs and their free
memory, the available memory) to choose:

- the decode and prefill thread counts and, where the CPUs allow it, which
  CPUs the decode threads use: the performance cluster alone, one thread
  pinned to each CPU, when one token streams the model's weight format faster
  there than on every CPU;
- the starting estimates the engine uses to place experts between the CPU and
  the GPU (it refines them as it runs);
- `vram.budget = 0` when the engine has CUDA and no CUDA GPU is usable, and on
  macOS the host memory budget from the runtime policy and a host margin of a
  quarter of the machine's memory (`host.headroom_mib`), since macOS compresses
  the cold pages of a RAM cache that leaves less free;
- with an engine that has CUDA and one usable CUDA GPU, which prompt passes the CPU prefills
  (`prefill.cpu_max_tokens`, the longest pass the CPU prefills faster than the
  GPU graph reading the same experts; a tie goes to the graph, which warms the
  VRAM tier for decode) and one prefill pass per prompt; on
  a GPU sharing host memory, a host margin of 4096 MiB (`host.headroom_mib`).

Only the prefill route and the pass size change the model's output, the way
any GPU kernel or batch shape does; the rest keeps it bit for bit. The launch fills only the settings
you left at their defaults: a value from `--set` or from the settings file always
wins, and a choice that contradicts one of your values is left out. Every choice and its reason is written to `strategy.json` in the run
directory and to the settings file the launch generates. Windows launches do
not use a profile yet.

## What is measured

| Section | Source | Content |
| --- | --- | --- |
| `cpu` | sysfs on Linux, `sysctl` on macOS; ISA from `inferssd-cpuprobe` | Logical CPUs with core, package, SMT siblings, `cpu_capacity` and maximum frequency; performance and efficiency clusters; whether threads can be pinned; the instruction-set features ggml reports (build options such as OpenMP or repacking are left out) |
| `memory` | `/proc/meminfo` or `hw.memsize`; `inferssd-cpuprobe` | Total memory, page size, whether the GPU shares system memory, DRAM read bandwidth per thread count |
| `compute` | `inferssd-cpuprobe` | ggml CPU matrix-multiply speed per weight format and thread count: weight bytes per second for one token, FLOP/s for more tokens; on Linux machines with several CPU clusters, also one token with a thread pinned to each CPU of each cluster and of every CPU (rows with `cpus`) |
| `gpus` | `inferssd-memprobe` (Linux with NVIDIA GPUs) | VRAM size, host-to-device (pinned and pageable), device-to-host and device-to-device copy rates, and device read bandwidth, for every GPU `nvidia-smi` lists |
| `storage:<drive>` | `inferssd-ssdprobe`, one per drive | Random-read bandwidth and latency by request size and queue depth, read with the engine's own reader; the knee queue depth per size |

The identity lists the GPUs, each with the backend the engine runs it on: the
NVIDIA GPUs `nvidia-smi` reports, on CUDA; on macOS the Apple GPU with its core
count as `ioreg` reports it, on Metal (when `ioreg` cannot say, the list is
empty and profiling continues); and on Linux the AMD and Intel display
controllers on the PCI bus, on Vulkan. Other display controllers, such as a
server's management display or a virtual machine's, are left out. The installer
builds the engine for the first of CUDA, Metal and Vulkan among these GPUs, and
for the CPU alone when there is none. The GPU probe measures every
GPU on Linux, regardless of `CUDA_VISIBLE_DEVICES`; on macOS the GPU shares the
machine's memory and is not measured. A GPU the probe cannot measure (too
little free VRAM, or a CUDA error) makes the `gpus` section partial, with the
GPU's UUID and the reason. When `nvidia-smi` is
installed but fails, for example after a driver upgrade without a reboot, the
`gpus` section fails and the saved GPU identity is kept, so the machine is not
mistaken for new hardware.

The CPU probe runs with one thread, the size of the performance cluster and
all logical CPUs. Clusters follow the engine's own rule: on Linux each CPU's
`cpu_capacity` (or, without it, `cpuinfo_max_freq`) is sorted in descending
order and split once at the largest relative drop above 10%; a CPU with
neither value makes the whole machine one cluster. On macOS the
`hw.perflevel0.logicalcpu` CPUs form the performance cluster and the rest the
efficiency cluster. macOS has no thread affinity, so its CPU numbers only count
CPUs.

The knee of a request size is the smallest queue depth whose best run reaches
the worst run of the fastest queue depth at that size. Each point's bandwidth is
the median of its runs.

Every number names its unit: `_bytes`, `_Bps` (bytes per second), `_us`, `_s`,
`_khz`, and `flops` (FLOP per second).

Facts that change while the machine runs are never stored: available memory,
free VRAM, running GPU processes and cgroup or affinity limits are read at
launch. Machine IDs and drive serial numbers are stored only as SHA-256 digests.

## Drive measurement

The SSD probe never writes. It reads random, aligned extents inside existing
files on the drive being measured, using direct I/O and the same reader the
engine uses (io_uring on Linux, a pool of reader threads on macOS). Model shards
are read as raw bytes, so no model format is involved. For each drive the
probe reads the eight largest files of at least 1 GiB among the given paths.

A given directory with no such file below it, such as an empty models directory
before the first download, has its drive measured in a test file instead:
`inferssd profile` writes 4 GiB of random data into it, past the page cache
(`F_NOCACHE` on macOS), has the probe read it, and removes it, also when the
probe fails. It needs 5 GiB free there. The file's name
(`.inferssd-drive-test-<pid>-*`) carries the run's process ID: test files are never
read as model files, and the next run removes those whose process has ended.
When no such directory is given, or there is not enough room, the drive's section
is recorded as failed with the reason.

A drive is identified by its model and a digest of its serial number, or by
its kernel name when it reports no serial (common for virtual disks). On Linux
the file's device is followed through partitions and device-mapper or md
layers to the physical disks; overlay and network filesystems have no disk and
are identified by filesystem type and mount point. On macOS the key uses the
model and protocol reported by `diskutil` and a digest of the APFS volume UUID;
a filesystem without a disk (network shares, FUSE) is identified by its type
and mount point.

## Speed estimates

With `--model`, `inferssd profile` estimates the decode speed of each supported
model, before it is downloaded, from this machine's profile and the facts of
its GGUF headers that the framework keeps. A token's time is the sum of:

- the dense weights a token reads, at the fastest measured memory read rate
  (the token embeddings and DeepSeek-V4.1's engram tables, of which a token
  reads one row, do not count);
- each routed expert at the CPU cost a launch prices one route at;
- the experts the RAM cache misses, read from the models' drive at the rate a
  launch takes for the engine's reads.

Memory available now, less the host margin a launch keeps (a quarter of memory
on macOS), holds the dense weights first, and what does not fit is read from the
drive each token; the RAM cache gets the rest and finds an expert as often as the
share of all experts that fits, as if routing were uniform. GPU time, prefetch
and the skew of real routing are not measured, so an estimate is a guide: on an
M5 MacBook Air (16 GB) and an M5 Max MacBook Pro (48 GB) the estimates came
within about a third of the measured decode speeds. Without a CPU measurement
of a model's expert formats or of its drive there is no estimate, and the
reason is printed instead.

## Where the file lives

| System | Default path |
| --- | --- |
| Linux | `$XDG_STATE_HOME/inferssd/machine.json`, by default `~/.local/state/inferssd/machine.json` |
| macOS | `~/Library/Application Support/inferssd/machine.json` |
| Windows | `%LOCALAPPDATA%\inferssd\machine.json` (reserved; profiling is not supported yet) |

The file is JSON with schema `inferssd-machine/1`. It is replaced atomically,
and a `machine.json.lock` file next to it keeps two profiles from running at
once. Each section records when it was measured, how long it took, the method,
its status (`ok`, `partial` or `failed`) and, when relevant, why.

## When sections are measured again

The saved file is reused section by section. A run measures:

- everything when there is no saved file, on `--reprofile`, or when the
  hardware identity differs: platform, architecture, machine ID, machine model,
  CPU brand, logical CPU count, GPUs, or total memory by more than 1%;
- the drives given with `--model` when the kernel or OS release changed (saved
  measurements of other drives are dropped, since they predate the change);
- `gpus` when the GPU driver changed;
- the sections of a probe whose output schema changed;
- a drive given with `--model` that the file does not describe yet;
- any section whose last attempt failed or was partial (for example a GPU
  that was busy).

`cpu`, `memory` and `compute` come from one CPU-probe run and are measured
together. When nothing needs measuring, the file is left untouched.

## How long it takes

Reading identity, topology and memory size takes well under a second. The
probes dominate, with these bounds from their default settings:

- The CPU probe times each format, thread count and token count for at least
  0.5 s; with seven formats, three thread counts and three token counts that is
  at least about 30 s, plus the DRAM reads.
- The SSD probe stops each point after 2 GiB or 5 s. With three request sizes,
  six queue depths and two repeats that is at most 36 points, about three
  minutes per drive on a slow drive and much less on a fast one.
- The GPU probe times copies of five sizes, from 256 KiB to 64 MiB, on each device.

Keep other heavy work off the machine while it is being profiled.
