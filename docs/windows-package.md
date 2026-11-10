# Precompiled Windows engine package

`Package Windows` builds the engine on GitHub's Windows runner. Users install the resulting ZIP;
their computers do not need Python, Git, Visual Studio, CMake, Ninja or the Vulkan SDK.
The desktop application consumes the same package through `scripts/install.ps1`.

The packages target Windows 11 x64 (build 22000 or later) and AVX2-capable CPUs. There is one
for each GPU backend the launch's table names on Windows:

- `windows-x64-cuda`, for NVIDIA GPUs: the CUDA backend with the GPU expert tier and the GPU
  memory probe, NVIDIA's CUDA runtime and cuBLAS libraries, built for Turing to Blackwell GPUs
  (real code for Ampere, Ada and Blackwell, PTX for the others). It needs an NVIDIA driver no
  older than its `minimum_nvidia_driver`, the driver of the CUDA toolkit it is built with.
  The driver's `nvcuda.dll` is loaded on first use, so the package also starts, on the CPU,
  where no NVIDIA driver is.
- `windows-x64-vulkan`, for every other machine: the Vulkan backend and loader, with CPU fallback when
  no usable GPU driver is present.

A GPU vendor's normal graphics driver is needed for GPU execution; the Vulkan SDK and the CUDA
toolkit are CI build dependencies only. The installer installs the CUDA package where an NVIDIA
GPU's driver is new enough, else the Vulkan package.

An archive has one directory, `inferssd-<date.build.commit>-windows-x64-vulkan` (or `-cuda`),
containing:

- `bin/inferssd.exe`: a relocatable native console entry point, forwarding arguments, standard
  handles and exit status to the package's own Python.
- `bin/inferssd-server.exe`, CPU/SSD probes (and the GPU memory probe with CUDA), engine DLLs,
  the Vulkan loader or the CUDA runtime, and Visual C++ runtimes.
- `python/`: pinned python-build-standalone CPython3.12 with the framework and hash-pinned Hub
  client dependencies. No package installation happens on the user's machine.
- `engine-resources.json`: paths relative to the package, including templates and probes.
- `templates/` and `licenses/`.

Beside the ZIP is an `inferssd-package 1` JSON listing with its size, SHA-256, `backend`,
Windows/CPU minimum (and NVIDIA driver minimum for CUDA), and the same build identity fields as
the macOS package. Published bytes are immutable.

## Build and verify

Every numbered `vX.Y.Z` release automatically calls the reusable Windows and macOS package
workflows at the same tag commit. After both platforms' checks pass, the common **Release** job
prepares one GitHub release containing the three archives (macOS, Windows Vulkan, Windows CUDA),
their manifests and `install.sh`/`install.ps1`, updates the normal latest package feeds, and only
then publishes the complete release. There is no
separate Windows-only version or prerelease channel. Publishing requires `APP_VERSIONING_TOKEN`.

The numbered Windows installer pins its default package URLs to that release's
`latest-windows-vulkan.json` and `latest-windows-cuda.json`; it cannot silently install a newer
rolling version. The asset preparation
step rejects packages with a different version, revision, build, label or checksum. Existing
published releases are never overwritten. A draft remains private until all assets and every
normal package feed are ready.

**Package Windows** still runs for packaging-related pull requests and can be started manually
for validation, producing `package-windows-x64-vulkan` and `package-windows-x64-cuda` Actions
artifacts. GitHub's runner has no GPU, so the CUDA package's check also shows that it starts
without an NVIDIA driver. Its optional manual publisher
remains main-only; no manual Windows job is required for normal numbered releases.

For a local build, enter an x64 Visual Studio tools environment, install the Vulkan SDK, and run:

```powershell
scripts/package-windows.ps1 -OutputDirectory dist
scripts/check-package-windows.ps1 -Archive dist/inferssd-<label>-windows-x64-vulkan.zip
```

For the CUDA package, install the CUDA toolkit instead (`scripts/install-cuda-redist.ps1` puts
NVIDIA's redistributable parts in a directory; set `CUDA_PATH` to it), enter the Visual Studio
tools environment with the host compiler that toolkit supports (`vcvars64.bat -vcvars_ver=14.44`
for CUDA 12.9), and pass `-Backend cuda`; its archive ends in `-windows-x64-cuda.zip`.

The check extracts under a different path containing spaces and removes developer tools from
PATH. It audits PE imports, loads the packaged Python's native dependencies, checks CLI/schema
and CPU execution, exercises Gemma's RAM-admission contract, and starts/stops an owned local
HTTP fixture through the packaged Windows guardian/job runtime. It then installs the ZIP using
the normal installer and verifies the installed command. No model weights are downloaded by
these packaging checks, and they do not claim GPU/model quality or throughput validation.

The main app installer is built separately in `wici-ai/wici-infer`; this repository publishes the
precompiled engine it downloads.
