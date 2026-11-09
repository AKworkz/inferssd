# SPDX-License-Identifier: Apache-2.0
# Usage: irm https://github.com/wici-ai/inferssd/releases/latest/download/install.ps1 | iex
# Install the precompiled Windows package without developer tools. A source build is an
# explicit developer option; either path keeps the working installation until checks pass.

# The release workflow replaces this with the tag it publishes.
$DefaultRef = 'main'
$Packages = 'https://appapi.wici.ai/versioning/inferssd/config/latest-windows.json'
$CudaPackages = 'https://appapi.wici.ai/versioning/inferssd/config/latest-windows-cuda.json'

function Install-InferSSD {
    param([string[]] $Arguments = @())
    $ErrorActionPreference = 'Stop'

    if ($Arguments -contains '--help') {
        @'
Install InferSSD for Windows x64 (Windows 11 / Windows Server build 22000 or later).
The default installation downloads a precompiled engine and its own Python runtime.
No Git, Python, compiler, Visual Studio or Vulkan SDK is needed. With an NVIDIA GPU
whose driver is new enough it installs the CUDA package; otherwise the Vulkan package,
which uses other GPUs through their drivers and also runs on the CPU.

Environment:
  INFERSSD_HOME         Installation and model root (default: %LOCALAPPDATA%\inferssd)
  INFERSSD_BIN_DIR      Command directory (default: %LOCALAPPDATA%\inferssd\bin)
  INFERSSD_PACKAGES     Prebuilt package list (default: the CUDA or Vulkan list, by GPU)
  INFERSSD_FROM_SOURCE  Build from source instead (any value, developer tools required)
  INFERSSD_REF          Explicit branch/tag to build from source
  INFERSSD_REPO         Git repository for an explicit source build
  INFERSSD_JOBS         Source-build parallelism (default: 4)
  INFERSSD_VCVARS       Source build's vcvars64.bat, if not discovered

No administrator rights are used. Models, settings and previous releases are preserved.
'@ | Write-Output
        return
    }
    if ($Arguments.Count) { throw 'inferssd: unknown argument; use --help.' }
    if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') { throw 'inferssd: Windows x64 is required.' }

    # cmd reads a batch file in the console's code page, the OEM one unless a user changed it.
    function Write-Batch([string] $Path, [string[]] $Lines) {
        $oem = [Text.Encoding]::GetEncoding([Globalization.CultureInfo]::CurrentCulture.TextInfo.OEMCodePage)
        [IO.File]::WriteAllLines($Path, $Lines, $oem)
    }

    $installHome = if ($env:INFERSSD_HOME) { $env:INFERSSD_HOME } else { Join-Path $env:LOCALAPPDATA 'inferssd' }
    $binDir = if ($env:INFERSSD_BIN_DIR) { $env:INFERSSD_BIN_DIR } else { Join-Path $env:LOCALAPPDATA 'inferssd\bin' }
    if (-not [IO.Path]::IsPathRooted($installHome)) { throw 'inferssd: INFERSSD_HOME must be an absolute path.' }
    if (-not [IO.Path]::IsPathRooted($binDir)) { throw 'inferssd: INFERSSD_BIN_DIR must be an absolute path.' }
    $installHome = [IO.Path]::GetFullPath($installHome).TrimEnd('\', '/')
    $binDir = [IO.Path]::GetFullPath($binDir).TrimEnd('\', '/')
    if ($installHome -match '^[a-zA-Z]:$') { $installHome += '\' }
    if ($binDir -match '^[a-zA-Z]:$') { $binDir += '\' }
    $releases = Join-Path $installHome 'releases'

    function Remove-Stage([string] $Path) {
        $absolute = [IO.Path]::GetFullPath($Path)
        if ([IO.Path]::GetDirectoryName($absolute) -ne $releases -or
            [IO.Path]::GetFileName($absolute) -notmatch '^(package|source)\.[a-zA-Z0-9]+$') {
            throw 'inferssd: refusing to clean a path outside the release staging directory.'
        }
        if (Test-Path -LiteralPath $absolute) {
            if ((Get-Item -LiteralPath $absolute -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw 'inferssd: refusing to recursively clean a reparse point.'
            }
            Remove-Item -LiteralPath $absolute -Recurse -Force
        }
    }

    function Remove-Link([string] $Path) {
        if (Test-Path -LiteralPath $Path) {
            if ([IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($Path)) -ne $installHome -or
                -not ((Get-Item -LiteralPath $Path -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
                throw 'inferssd: the installation pointer is not a junction.'
            }
            # Non-recursive deletion removes the junction, never the release it names.
            [IO.Directory]::Delete($Path)
        }
    }

    function Publish-Release([string] $Package, [string] $Command, [string] $Resources) {
        [IO.Directory]::CreateDirectory($binDir) | Out-Null
        $id = [Guid]::NewGuid().ToString('N')
        $current = Join-Path $installHome 'current'
        $pending = Join-Path $installHome ('.current-' + $id)
        $previous = Join-Path $installHome ('.previous-' + $id)
        $wrapper = Join-Path $binDir 'inferssd.cmd'
        $pendingCommand = Join-Path $binDir ('.inferssd-' + $id + '.cmd')
        $previousCommand = Join-Path $binDir ('.previous-' + $id + '.cmd')
        $moved = $false; $published = $false; $commandPublished = $false; $finished = $false
        Write-Batch $pendingCommand @(
            '@echo off',
            "if not defined INFERSSD_HOME set `"INFERSSD_HOME=$installHome`"",
            "if not defined INFERSSD_RESOURCES set `"INFERSSD_RESOURCES=$Resources`"",
            "`"$Command`" %*"
        )
        New-Item -ItemType Junction -Path $pending -Value $Package | Out-Null
        try {
            if (Test-Path -LiteralPath $current) {
                if (-not ((Get-Item -LiteralPath $current -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
                    throw 'inferssd: current is not an installation junction; it was left unchanged.'
                }
                [IO.Directory]::Move($current, $previous)
                $moved = $true
            }
            [IO.Directory]::Move($pending, $current)
            $published = $true
            if (Test-Path -LiteralPath $wrapper) {
                [IO.File]::Replace($pendingCommand, $wrapper, $previousCommand)
            } else {
                [IO.File]::Move($pendingCommand, $wrapper)
            }
            $commandPublished = $true
            $finished = $true
        } catch {
            $failure = $_
            if ($published) { Remove-Link $current }
            if ($moved) { [IO.Directory]::Move($previous, $current) }
            if ($commandPublished -and (Test-Path -LiteralPath $previousCommand)) {
                [IO.File]::Replace($previousCommand, $wrapper, $null)
            }
            throw $failure
        } finally {
            Remove-Link $pending
            if ($finished) { Remove-Link $previous }
            foreach ($file in @($pendingCommand, $previousCommand)) {
                if (Test-Path -LiteralPath $file) { Remove-Item -LiteralPath $file -Force }
            }
        }
    }

    function Download-File([string] $Address, [string] $Destination) {
        $uri = [Uri]$Address
        if (-not $uri.IsAbsoluteUri -or $uri.UserInfo -or $uri.Fragment -or
            ($uri.Scheme -ne 'https' -and -not ($uri.IsFile -and $env:INFERSSD_PACKAGES) -and
             -not ($uri.Scheme -eq 'http' -and $uri.Host -in @('127.0.0.1', 'localhost', '[::1]')))) {
            throw 'inferssd: package URLs require HTTPS (or a local test package list).'
        }
        if ($uri.IsFile) { Copy-Item -LiteralPath $uri.LocalPath -Destination $Destination }
        else {
            # Windows PowerShell 5 redraws its progress bar for every block it receives, which makes a
            # download of hundreds of MB take minutes even from this machine.
            $ProgressPreference = 'SilentlyContinue'
            Invoke-WebRequest -UseBasicParsing -Uri $uri.AbsoluteUri -OutFile $Destination
        }
    }

    # The NVIDIA driver version of this machine's NVIDIA GPU, or $null. PCI vendor 10DE is the CUDA
    # row of the backend table the launch reads (inferssd.machine.identity.PCI_BACKENDS).
    function Get-NvidiaDriver {
        foreach ($adapter in @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue)) {
            if ([string]$adapter.PNPDeviceID -match '^PCI\\VEN_10DE&' -and
                [string]$adapter.DriverVersion -match '^[0-9]+\.[0-9]+\.([0-9]+)\.([0-9]+)$') {
                # Windows writes NVIDIA's driver 591.91 as 32.0.15.9191: its last five digits.
                $digits = $Matches[1] + $Matches[2].PadLeft(4, '0')
                $digits = $digits.Substring($digits.Length - 5)
                return [Version]($digits.Substring(0, 3) + '.' + $digits.Substring(3))
            }
        }
        return $null
    }

    function Get-PackageList([string] $Address, [string] $Destination) {
        try { Download-File $Address $Destination }
        catch { throw "inferssd: cannot download the Windows package list. No source build was started. $($_.Exception.Message)" }
        return Get-Content -LiteralPath $Destination -Raw | ConvertFrom-Json
    }

    function Install-Package {
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        [IO.Directory]::CreateDirectory($releases) | Out-Null
        # Keep the installed path short enough for Windows PowerShell 5 / .NET Framework.
        $release = Join-Path $releases ('package.' + [Guid]::NewGuid().ToString('N').Substring(0,12))
        [IO.Directory]::CreateDirectory($release) | Out-Null
        $complete = $false
        try {
            $listing = Join-Path $release 'latest.json'
            if ($env:INFERSSD_PACKAGES) {
                $package = Get-PackageList $env:INFERSSD_PACKAGES $listing
            } else {
                # An NVIDIA GPU runs the CUDA package once its driver runs the package's CUDA runtime.
                $driver = Get-NvidiaDriver
                $package = if ($driver) { Get-PackageList $CudaPackages $listing }
                if ($package -and $driver -lt [Version]$package.minimum_nvidia_driver) {
                    Write-Output ("inferssd: the CUDA package needs NVIDIA driver $($package.minimum_nvidia_driver) or later " +
                        "and this machine has $driver; installing the Vulkan package. Update the driver and run " +
                        'this installer again to use CUDA.')
                    $package = $null
                }
                if (-not $package) { $package = Get-PackageList $Packages $listing }
            }
            # Package lists from before the CUDA package carry no backend: they are Vulkan's.
            $backend = if ($package.backend) { $package.backend } else { 'vulkan' }
            $suffix = @{ vulkan = 'windows-x64'; cuda = 'windows-x64-cuda' }[[string]$backend]
            if ($package.schema -ne 'inferssd-package 1' -or $package.platform -ne 'windows-x64' -or -not $suffix) {
                throw 'inferssd: unsupported Windows package list.'
            }
            if ($package.name -notmatch ('^inferssd-[a-zA-Z0-9.-]+-' + $suffix + '$') -or
                $package.file -ne ($package.name + '.zip') -or $package.sha256 -notmatch '^[a-fA-F0-9]{64}$' -or
                ($package.size -isnot [int] -and $package.size -isnot [long]) -or [long]$package.size -lt 1 -or
                ($package.build -isnot [int] -and $package.build -isnot [long]) -or [long]$package.build -lt 1 -or
                $package.revision -notmatch '^[a-fA-F0-9]{40}$' -or
                $package.minimum_windows -notmatch '^10\.0\.[0-9]+(?:\.[0-9]+)?$' -or
                ($backend -eq 'cuda' -and [string]$package.minimum_nvidia_driver -notmatch '^[0-9]+\.[0-9]+$')) {
                throw 'inferssd: invalid Windows package metadata.'
            }
            if ([Environment]::OSVersion.Version -lt [Version]$package.minimum_windows) {
                throw "inferssd: this package requires Windows build $($package.minimum_windows) or later."
            }
            Write-Output "Downloading InferSSD $($package.label), $backend ($([math]::Ceiling($package.size / 1000000)) MB)."
            $archive = Join-Path $release 'package.zip'
            Download-File $package.url $archive
            $hash = [Security.Cryptography.SHA256]::Create()
            $stream = [IO.File]::OpenRead($archive)
            try { $digest = [BitConverter]::ToString($hash.ComputeHash($stream)).Replace('-', '') }
            finally { $hash.Dispose(); $stream.Dispose() }
            if ((Get-Item -LiteralPath $archive).Length -ne [long]$package.size -or
                $digest -ne $package.sha256) {
                throw 'inferssd: the package download does not match its size and checksum.'
            }
            $zip = [IO.Compression.ZipFile]::OpenRead($archive)
            try {
                function Entry-Path($entry) {
                    $relative = $entry.FullName.Replace('/', '\')
                    $link = (($entry.ExternalAttributes -shr 16) -band 61440) -eq 40960
                    if ($relative.Contains(':') -or $link -or
                        -not $relative.StartsWith($package.name + '\', [StringComparison]::Ordinal)) {
                        throw 'inferssd: unsafe path in the Windows package.'
                    }
                    $relative = $relative.Substring($package.name.Length + 1)
                    if (!$relative -and !$entry.Length) { return $null }
                    # The archive's validated single top folder is not duplicated on disk.
                    $target = [IO.Path]::GetFullPath((Join-Path $release $relative))
                    if (!$target.StartsWith($release + '\', [StringComparison]::OrdinalIgnoreCase)) {
                        throw 'inferssd: unsafe path in the Windows package.'
                    }
                    return $target
                }
                foreach ($entry in $zip.Entries) { [void](Entry-Path $entry) }
                Write-Output "Unpacking InferSSD $($package.label)."
                foreach ($entry in $zip.Entries) {
                    $target = Entry-Path $entry
                    if (!$target) { continue }
                    if ($entry.FullName.EndsWith('/') -or $entry.FullName.EndsWith('\')) {
                        [IO.Directory]::CreateDirectory($target) | Out-Null
                    } else {
                        [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($target)) | Out-Null
                        [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $target)
                    }
                }
            } finally { $zip.Dispose() }
            $root = $release
            $command = Join-Path $root 'bin\inferssd.exe'
            $server = Join-Path $root 'bin\inferssd-server.exe'
            $resources = Join-Path $root 'engine-resources.json'
            foreach ($file in @($command, $server, $resources, (Join-Path $root 'python\python.exe'))) {
                if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw 'inferssd: the package is incomplete.' }
            }
            $oldResources = $env:INFERSSD_RESOURCES
            try {
                $env:INFERSSD_RESOURCES = $resources
                $version = & $command --version --json
                if ($LASTEXITCODE) { throw 'inferssd: the packaged command does not run.' }
                $version = $version | ConvertFrom-Json
                if ($version.build -ne $package.build -or $version.revision -ne $package.revision) {
                    throw 'inferssd: the command version differs from its package list.'
                }
                & $command cli --help | Out-Null
                if ($LASTEXITCODE) { throw 'inferssd: the packaged chat command does not run.' }
                & $command serve --help | Out-Null
                if ($LASTEXITCODE) { throw 'inferssd: the packaged server command does not run.' }
                & $server --ssd-schema | Out-Null
                if ($LASTEXITCODE) { throw 'inferssd: the native engine or its runtime libraries do not run.' }
            } finally {
                if ($null -eq $oldResources) { Remove-Item Env:INFERSSD_RESOURCES -ErrorAction SilentlyContinue }
                else { $env:INFERSSD_RESOURCES = $oldResources }
            }
            Remove-Item -LiteralPath $archive, $listing -Force
            # The verified package remains available even if publishing the pointer fails.
            $complete = $true
            Publish-Release $root $command $resources
            Write-Output "Installed: $binDir\inferssd.cmd"
            if (($env:PATH -split ';') -notcontains $binDir) {
                Write-Output 'Add this directory to your terminal PATH:'
                Write-Output "  `$env:PATH = '$binDir;' + `$env:PATH"
            }
        } finally {
            if (-not $complete) { Remove-Stage $release }
        }
    }

    if (-not $env:INFERSSD_FROM_SOURCE -and -not $env:INFERSSD_REF) {
        Install-Package
        return
    }

    function Require([string] $Name) {
        if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
            throw "inferssd: install $Name first. See docs/quickstart.md."
        }
    }
    Require git

    # The Microsoft Store's python.exe only opens the Store, so it must also run a program.
    $python = $null
    foreach ($candidate in @(@('python'), @('py', '-3'), @('python3'))) {
        if (Get-Command $candidate[0] -ErrorAction SilentlyContinue) {
            $rest = @($candidate | Select-Object -Skip 1)
            & $candidate[0] @rest -c 'import sys; sys.exit(sys.version_info < (3, 9))' 2>$null
            if ($LASTEXITCODE -eq 0) { $python = $candidate; break }
        }
    }
    if (-not $python) { throw 'inferssd: Python >=3.9 is required.' }

    $llvm = $null
    if (Get-Command clang -ErrorAction SilentlyContinue) {
        $llvm = Split-Path (Get-Command clang).Source
    } elseif (Test-Path "$env:ProgramFiles\LLVM\bin\clang.exe") {
        $llvm = "$env:ProgramFiles\LLVM\bin"
    } else {
        throw 'inferssd: install LLVM clang first. See docs/quickstart.md.'
    }

    $vcvars = $env:INFERSSD_VCVARS
    if (-not $vcvars) {
        $where = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
        if (Test-Path $where) {
            $found = & $where -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
            if ($found) { $vcvars = Join-Path $found 'VC\Auxiliary\Build\vcvars64.bat' }
        }
    }
    if (-not $vcvars -or -not (Test-Path $vcvars)) {
        throw 'inferssd: install the Visual Studio C++ build tools, or set INFERSSD_VCVARS to vcvars64.bat.'
    }

    $jobs = if ($env:INFERSSD_JOBS) { $env:INFERSSD_JOBS } else { '4' }
    if ($jobs -notmatch '^[1-9][0-9]*$') { throw 'inferssd: INFERSSD_JOBS must be a positive integer.' }
    $ref = if ($env:INFERSSD_REF) { $env:INFERSSD_REF } else { $DefaultRef }
    $repository = if ($env:INFERSSD_REPO) { $env:INFERSSD_REPO } else { 'https://github.com/wici-ai/inferssd.git' }

    New-Item -ItemType Directory -Force (Join-Path $installHome 'releases'), $binDir | Out-Null
    $release = Join-Path $installHome ('releases\source.' + [IO.Path]::GetRandomFileName().Split('.')[0])
    New-Item -ItemType Directory $release | Out-Null
    $complete = $false
    try {
        $source = Join-Path $release 'source'
        # Keep the history for engine-resources.py's build number, without old file contents.
        & git clone --filter=blob:none --single-branch --branch $ref $repository $source
        if ($LASTEXITCODE) { throw 'inferssd: git clone failed.' }
        $rest = @($python | Select-Object -Skip 1)
        # The GPU backend for this machine's GPUs, by the table the launch reads: CUDA for an NVIDIA
        # GPU, Vulkan for the others.
        $env:PYTHONPATH = Join-Path $source 'framework\src'
        $backend = & $python[0] @rest -m inferssd.machine.backend
        $found = $LASTEXITCODE
        Remove-Item Env:PYTHONPATH
        if ($found) { throw 'inferssd: reading the GPUs failed.' }
        $preset = 'windows-llvm'
        $vulkan = ''
        $targets = 'inferssd-server inferssd-cpuprobe inferssd-ssdprobe'
        if ($backend -eq 'CUDA') {
            if (-not $env:CUDA_PATH -or -not (Test-Path -LiteralPath (Join-Path $env:CUDA_PATH 'bin\nvcc.exe'))) {
                throw 'inferssd: an NVIDIA GPU was found: install the CUDA toolkit (it sets CUDA_PATH).'
            }
            $preset = 'windows-cuda'
            $targets += ' inferssd-memprobe'
        } elseif ($backend -ne 'CPU') {
            if ($env:VULKAN_SDK) { $vulkan = ' -DGGML_VULKAN=ON' }
            else { Write-Output 'inferssd: a GPU was found; install the Vulkan SDK to use it.' }
        }
        $kind = if ($vulkan) { "$preset, Vulkan" } else { $preset }
        Write-Output "Building InferSSD ($kind). This may take several minutes."
        & $python[0] @rest -m venv (Join-Path $release 'venv')
        if ($LASTEXITCODE) { throw 'inferssd: creating the Python environment failed.' }
        $scripts = Join-Path $release 'venv\Scripts'
        $py = Join-Path $scripts 'python.exe'
        & $py -m pip install --upgrade pip
        if ($LASTEXITCODE) { throw 'inferssd: updating pip failed.' }
        & $py -m pip install 'cmake>=3.24,<5' ninja "$source\framework[hf]"
        if ($LASTEXITCODE) { throw 'inferssd: installing the framework failed.' }

        $build = Join-Path $release 'build.bat'
        # A batch file: only cmd can keep vcvars64.bat's environment for the commands after it.
        Write-Batch $build @(
            '@echo off',
            "call `"$vcvars`" > nul || exit /b 1",
            "set `"PATH=$scripts;$llvm;%VULKAN_SDK%\Bin;%CUDA_PATH%\bin;%PATH%`"",
            "cd /d `"$source`" || exit /b 1",
            "cmake --preset $preset$vulkan || exit /b 1",
            "cmake --build build-$preset --parallel $jobs --target $targets || exit /b 1"
        )
        & cmd /c $build
        if ($LASTEXITCODE) { throw 'inferssd: building the engine failed.' }
        $resources = Join-Path $release 'engine-resources.json'
        & $py (Join-Path $source 'scripts\engine-resources.py') --build (Join-Path $source "build-$preset") --output $resources
        if ($LASTEXITCODE) { throw 'inferssd: describing the engine failed.' }
        $inferssd = Join-Path $scripts 'inferssd.exe'
        & $inferssd cli --help | Out-Null
        if ($LASTEXITCODE) { throw 'inferssd: the installed command does not run.' }
        & $inferssd serve --help | Out-Null
        if ($LASTEXITCODE) { throw 'inferssd: the installed command does not run.' }
        # Everything below may publish a pointer to this fully verified release.
        # Retain it even if publishing the command fails halfway through.
        $complete = $true

        Publish-Release $release $inferssd $resources
    } finally {
        if (-not $complete) { Remove-Stage $release }
    }

    Write-Output ''
    Write-Output "Installed: $binDir\inferssd.cmd"
    if (($env:PATH -split ';') -notcontains $binDir) {
        Write-Output 'Add this directory to your PATH (run now, and add it to your user environment):'
        Write-Output "  `$env:PATH = '$binDir;' + `$env:PATH"
    }
    @'

Download and chat (first launch downloads hundreds of GB):
  inferssd cli -hf unsloth/Qwen3.8-Flash-Next-GGUF

Start the OpenAI-compatible API:
  inferssd serve -hf unsloth/Qwen3.8-Flash-Next-GGUF
'@ | Write-Output
}

Install-InferSSD -Arguments $args
