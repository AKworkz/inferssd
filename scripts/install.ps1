# SPDX-License-Identifier: Apache-2.0
# Usage: irm https://github.com/wici-ai/inferssd/releases/latest/download/install.ps1 | iex
# The same installation as install.sh: a new release directory built from source, published
# only once it works, and a command that uses it. A previous installation stays usable.

# The release workflow replaces this with the tag it publishes.
$DefaultRef = 'main'

function Install-InferSSD {
    param([string[]] $Arguments = @())
    $ErrorActionPreference = 'Stop'

    if ($Arguments -contains '--help') {
        @'
Install InferSSD from source for Windows x64.
Requires Git, Python >=3.9, LLVM clang and the Visual Studio C++ build tools with a
Windows SDK of 10.0.22000 or later. The engine is built for the machine's GPUs: with
the Vulkan SDK (VULKAN_SDK) it uses a GPU through Vulkan, an NVIDIA GPU included;
without a GPU or the Vulkan SDK it runs on the CPU.
CMake, Ninja and the Python framework are installed in a private environment.

Environment:
  INFERSSD_HOME         Installation and model root (default: %LOCALAPPDATA%\inferssd)
  INFERSSD_BIN_DIR      Command directory (default: %LOCALAPPDATA%\inferssd\bin)
  INFERSSD_REF          Branch or tag to build (default: this installer's release)
  INFERSSD_REPO         Git repository to build from (default: the InferSSD repository)
  INFERSSD_JOBS         Build parallelism (default: 4)
  INFERSSD_VCVARS       vcvars64.bat, if Visual Studio's installer does not know it

No administrator rights are used. A previous installation remains usable if building fails.
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

    $installHome = if ($env:INFERSSD_HOME) { $env:INFERSSD_HOME } else { Join-Path $env:LOCALAPPDATA 'inferssd' }
    $binDir = if ($env:INFERSSD_BIN_DIR) { $env:INFERSSD_BIN_DIR } else { Join-Path $env:LOCALAPPDATA 'inferssd\bin' }
    if (-not [IO.Path]::IsPathRooted($installHome)) { throw 'inferssd: INFERSSD_HOME must be an absolute path.' }
    if (-not [IO.Path]::IsPathRooted($binDir)) { throw 'inferssd: INFERSSD_BIN_DIR must be an absolute path.' }
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
        & git clone --depth 1 --branch $ref $repository $source
        if ($LASTEXITCODE) { throw 'inferssd: git clone failed.' }
        $rest = @($python | Select-Object -Skip 1)
        # The GPU backend for this machine's GPUs, by the table the launch reads. This installer
        # builds a GPU's engine with Vulkan, an NVIDIA GPU's too.
        $env:PYTHONPATH = Join-Path $source 'framework\src'
        $backend = & $python[0] @rest -m inferssd.machine.backend
        $found = $LASTEXITCODE
        Remove-Item Env:PYTHONPATH
        if ($found) { throw 'inferssd: reading the GPUs failed.' }
        $vulkan = ''
        if ($backend -ne 'CPU') {
            if ($env:VULKAN_SDK) { $vulkan = ' -DGGML_VULKAN=ON' }
            else { Write-Output 'inferssd: a GPU was found; install the Vulkan SDK to use it.' }
        }
        $kind = if ($vulkan) { 'windows-llvm, Vulkan' } else { 'windows-llvm' }
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
            "set `"PATH=$scripts;$llvm;%VULKAN_SDK%\Bin;%PATH%`"",
            "cd /d `"$source`" || exit /b 1",
            "cmake --preset windows-llvm$vulkan || exit /b 1",
            "cmake --build build-windows-llvm --parallel $jobs --target inferssd-server inferssd-cpuprobe inferssd-ssdprobe || exit /b 1"
        )
        & cmd /c $build
        if ($LASTEXITCODE) { throw 'inferssd: building the engine failed.' }
        $resources = Join-Path $release 'engine-resources.json'
        & $py (Join-Path $source 'scripts\engine-resources.py') --build (Join-Path $source 'build-windows-llvm') --output $resources
        if ($LASTEXITCODE) { throw 'inferssd: describing the engine failed.' }
        $inferssd = Join-Path $scripts 'inferssd.exe'
        & $inferssd cli --help | Out-Null
        if ($LASTEXITCODE) { throw 'inferssd: the installed command does not run.' }
        & $inferssd serve --help | Out-Null
        if ($LASTEXITCODE) { throw 'inferssd: the installed command does not run.' }
        # Everything below may publish a pointer to this fully verified release.
        # Retain it even if publishing the command fails halfway through.
        $complete = $true

        $pending = Join-Path $binDir ('.inferssd-' + [IO.Path]::GetRandomFileName() + '.cmd')
        Write-Batch $pending @(
            '@echo off',
            "if not defined INFERSSD_HOME set `"INFERSSD_HOME=$installHome`"",
            "if not defined INFERSSD_RESOURCES set `"INFERSSD_RESOURCES=$resources`"",
            "`"$inferssd`" %*"
        )
        $current = Join-Path $installHome 'current'
        if (Test-Path $current) { & cmd /c rmdir "$current" }
        & cmd /c mklink /J "$current" "$release" | Out-Null
        if ($LASTEXITCODE) { throw 'inferssd: publishing the release failed.' }
        Move-Item -Force $pending (Join-Path $binDir 'inferssd.cmd')
    } finally {
        if (-not $complete) { Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $release }
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
