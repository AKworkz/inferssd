param(
  # The GPU backend the engine is built with, by the table the launch reads: CUDA for NVIDIA GPUs,
  # Vulkan for the others (and the CPU where there is no GPU).
  [ValidateSet('vulkan', 'cuda')][string]$Backend = 'vulkan',
  [string]$OutputDirectory = (Join-Path $PSScriptRoot '..\dist'),
  [string]$BuildDirectory = '',
  [int]$Jobs = 2,
  # Real code for the GPU generations people run; PTX for the rest, compiled by the driver on first use.
  [string]$CudaArchitectures = '75-virtual;80-virtual;86-real;89-real;120a-real',
  # The driver of the CUDA toolkit this package is built with (12.9 Update 1): the oldest that runs its
  # runtime and compiles its PTX.
  [string]$MinimumNvidiaDriver = '576.57'
)
# Build tools are used here, in CI, never by the machine installing the resulting ZIP.
$ErrorActionPreference = 'Stop'
if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') { throw 'Package Windows on Windows x64.' }
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$out = [IO.Path]::GetFullPath($OutputDirectory)
if (!$BuildDirectory) { $BuildDirectory = Join-Path $PSScriptRoot "..\build-package-windows-$Backend" }
$build = [IO.Path]::GetFullPath($BuildDirectory)
$suffix = if ($Backend -eq 'cuda') { 'windows-x64-cuda' } else { 'windows-x64' }
if ((& git -C $root rev-parse --is-shallow-repository).Trim() -ne 'false') { throw 'Packaging needs full Git history for the build number.' }
if ($Backend -eq 'vulkan' -and (!$env:VULKAN_SDK -or !(Test-Path -LiteralPath (Join-Path $env:VULKAN_SDK 'Bin\glslc.exe')))) { throw 'Set VULKAN_SDK to the build-time SDK.' }
if ($Backend -eq 'cuda' -and (!$env:CUDA_PATH -or !(Test-Path -LiteralPath (Join-Path $env:CUDA_PATH 'bin\nvcc.exe')))) { throw 'Set CUDA_PATH to the build-time CUDA toolkit.' }
function Native([string]$Command, [string[]]$Arguments) {
  & $Command @Arguments
  if ($LASTEXITCODE -ne 0) { throw "$Command failed ($LASTEXITCODE)." }
}
function Download([string]$Url, [string]$Path, [string]$Sha256) {
  Invoke-WebRequest -UseBasicParsing -UserAgent 'inferssd-build' -Uri $Url -OutFile $Path
  if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -ne $Sha256) { throw "Download checksum mismatch: $Url" }
}
$temporary = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\')
# A short name: Windows PowerShell 5's .NET refuses paths of 260 characters or more, and the archive's
# deepest file (a __pycache__ entry of huggingface_hub) sits about 200 characters below it.
$stage = Join-Path $temporary ('ip-' + [Guid]::NewGuid().ToString('N').Substring(0, 12))
$package = Join-Path $stage 'package'
[IO.Directory]::CreateDirectory($package) | Out-Null
function Remove-StagedDirectory([string]$Path) {
  $absolute = [IO.Path]::GetFullPath($Path)
  if (!$absolute.StartsWith($package + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Refusing cleanup outside the package staging directory.' }
  if (Test-Path -LiteralPath $absolute) { Remove-Item -LiteralPath $absolute -Recurse -Force }
}
try {
  $configure = @('-S', $root, '-B', $build, '-DGGML_NATIVE=OFF', '-DGGML_BACKEND_DL=OFF',
    '-DGGML_CPU_ALL_VARIANTS=OFF', '-DGGML_OPENMP=OFF', '-DLLAMA_OPENSSL=OFF', '-DLLAMA_BUILD_TESTS=OFF')
  $targets = @('inferssd-server', 'inferssd-cpuprobe', 'inferssd-ssdprobe')
  if ($Backend -eq 'cuda') {
    $cuda = $env:CUDA_PATH.Replace('\', '/')
    $configure = @('--preset', 'windows-cuda') + $configure + @('-DGGML_VULKAN=OFF',
      "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures", "-DCUDAToolkit_ROOT=$cuda", "-DCMAKE_CUDA_COMPILER=$cuda/bin/nvcc.exe")
    $targets += 'inferssd-memprobe'
  } else {
    $configure = @('--preset', 'windows-llvm') + $configure + @('-DGGML_VULKAN=ON', '-DGGML_CUDA=OFF')
  }
  Native cmake $configure
  Native cmake (@('--build', $build, '--parallel', [string]$Jobs, '--target') + $targets)
  foreach ($folder in 'bin', 'templates', 'licenses') { [IO.Directory]::CreateDirectory((Join-Path $package $folder)) | Out-Null }
  Get-ChildItem -LiteralPath (Join-Path $build 'bin') -Filter '*.dll' | Copy-Item -Destination (Join-Path $package 'bin')
  foreach ($name in $targets) {
    Copy-Item -LiteralPath (Join-Path $build "bin\$name.exe") -Destination (Join-Path $package 'bin')
  }
  # The GPU runtime is redistributable; no SDK or GPU development tools are installed by users.
  # Descriptor generation validates the original build-side server too, so it gets the runtime as well.
  if ($Backend -eq 'cuda') {
    # The runtime and cuBLAS the engine imports; nvcuda.dll comes with the user's NVIDIA driver.
    $runtimeFiles = @(foreach ($pattern in 'cudart64_*.dll', 'cublas64_*.dll', 'cublasLt64_*.dll') {
      Get-ChildItem -LiteralPath (Join-Path $env:CUDA_PATH 'bin') -Filter $pattern })
    if ($runtimeFiles.Count -ne 3) { throw 'The CUDA runtime and cuBLAS libraries were not found in CUDA_PATH.' }
    $runtimeFiles | Copy-Item -Destination (Join-Path $package 'bin')
    $runtimeFiles | Copy-Item -Destination (Join-Path $build 'bin') -Force
    Copy-Item -LiteralPath (Join-Path $env:CUDA_PATH 'LICENSE') -Destination (Join-Path $package 'licenses\LICENSE-CUDA.txt')
  } else {
    $runtime = Join-Path $stage 'vulkan-runtime.zip'
    Download 'https://sdk.lunarg.com/sdk/download/1.4.363.0/windows/VulkanRT-X64-1.4.363.0-Components.zip' $runtime 'a25a927aa8b9f0371048f1861cf88ac3b9bc9b1fb332c42d897c8ab32695769a'
    Expand-Archive -LiteralPath $runtime -DestinationPath (Join-Path $stage 'vulkan-runtime')
    $runtimeRoot = Join-Path $stage 'vulkan-runtime\VulkanRT-X64-1.4.363.0-Components'
    Copy-Item -LiteralPath (Join-Path $runtimeRoot 'x64\vulkan-1.dll') -Destination (Join-Path $package 'bin')
    Copy-Item -LiteralPath (Join-Path $runtimeRoot 'x64\vulkan-1.dll') -Destination (Join-Path $build 'bin')
    Copy-Item -LiteralPath (Join-Path $runtimeRoot 'VulkanRT-License.txt') -Destination (Join-Path $package 'licenses')
  }

  $pythonArchive = Join-Path $stage 'python.tar.gz'
  Download 'https://github.com/astral-sh/python-build-standalone/releases/download/20261026/cpython-3.12.16%2B20261026-x86_64-pc-windows-msvc-install_only_stripped.tar.gz' $pythonArchive '1f2ab53880fc67ea323398c6388e05af28addc99a8e389209f08a51fdff4b149'
  Native tar @('-xzf', $pythonArchive, '-C', $package)
  $python = Join-Path $package 'python\python.exe'
  Native $python @('-I', '-X', 'utf8', '-m', 'pip', '--disable-pip-version-check', '--no-cache-dir', 'install', '--no-warn-script-location', '--only-binary=:all:', '--require-hashes', '-r', (Join-Path $PSScriptRoot 'package-windows-requirements.txt'))
  Native $python @('-I', '-X', 'utf8', '-m', 'pip', '--disable-pip-version-check', '--no-cache-dir', 'install', '--no-warn-script-location', '--no-deps', '--no-build-isolation', '--no-index', (Join-Path $root 'framework'))
  Native $python @('-I', '-m', 'pip', 'uninstall', '-y', '-q', 'pip', 'setuptools')
  Remove-StagedDirectory (Join-Path $package 'python\Scripts')
  Remove-StagedDirectory (Join-Path $package 'python\Lib\ensurepip')
  Remove-StagedDirectory (Join-Path $package 'python\Lib\test')
  Remove-StagedDirectory (Join-Path $package 'python\include')

  $crt = if ($env:VCToolsRedistDir) { Join-Path $env:VCToolsRedistDir 'x64' } else { '' }
  $crtFiles = if ($crt -and (Test-Path -LiteralPath $crt)) { @(Get-ChildItem -LiteralPath $crt -Recurse -Filter '*.dll' | Where-Object {$_.Directory.Name -match '^Microsoft\.VC[0-9]+\.CRT$'}) } else { @() }
  if (!$crtFiles.Count) { throw 'The Visual C++ redistributable CRT files were not found; run from vcvars64.bat.' }
  foreach ($folder in 'bin', 'python') { $crtFiles | Copy-Item -Destination (Join-Path $package $folder) -Force }
  $crtFiles | Copy-Item -Destination (Join-Path $build 'bin') -Force
  Native clang++ @('--target=x86_64-pc-windows-msvc', '-std=c++17', '-O2', '-DUNICODE', '-D_UNICODE', (Join-Path $PSScriptRoot 'windows-launcher.cpp'), '-o', (Join-Path $package 'bin\inferssd.exe'))
  Native $python @('-I', (Join-Path $PSScriptRoot 'engine-resources.py'), '--build', $build, '--output', (Join-Path $stage 'resources.json'))
  $resource = Get-Content -LiteralPath (Join-Path $stage 'resources.json') -Raw | ConvertFrom-Json
  if ($env:GITHUB_ACTIONS -eq 'true') {
    # CI checks out the immutable event SHA; retain the branch name as descriptive metadata.
    $resource.branch = if ($env:GITHUB_HEAD_REF) { $env:GITHUB_HEAD_REF } else { $env:GITHUB_REF_NAME }
  }
  $resource.binary = 'bin/inferssd-server.exe'
  foreach ($property in @($resource.templates.PSObject.Properties)) {
    $file = Get-Item -LiteralPath $property.Value
    Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $package 'templates')
    $property.Value = 'templates/' + $file.Name
  }
  foreach ($property in @($resource.probes.PSObject.Properties)) { $property.Value = 'bin/' + [IO.Path]::GetFileName($property.Value) }
  $utf8 = [Text.UTF8Encoding]::new($false)
  [IO.File]::WriteAllText((Join-Path $package 'engine-resources.json'), ($resource | ConvertTo-Json -Depth 8), $utf8)
  Copy-Item -LiteralPath (Join-Path $root 'LICENSE'), (Join-Path $root 'NOTICE') -Destination (Join-Path $package 'licenses')
  Copy-Item -LiteralPath (Join-Path $root 'engine\LICENSE') -Destination (Join-Path $package 'licenses\LICENSE-llama.cpp')
  Get-ChildItem -LiteralPath (Join-Path $root 'engine\licenses') -File | Copy-Item -Destination (Join-Path $package 'licenses')
  $command = Join-Path $package 'bin\inferssd.exe'
  $previousResources = $env:INFERSSD_RESOURCES
  try {
    $env:INFERSSD_RESOURCES = Join-Path $package 'engine-resources.json'
    $version = (& $command --version --json | ConvertFrom-Json)
    if ($LASTEXITCODE -ne 0 -or !$version.label) { throw 'The packaged command has no valid build label.' }
    Native $command @('serve', '--help')
  } finally { $env:INFERSSD_RESOURCES = $previousResources }
  $name = 'inferssd-' + $version.label + '-' + $suffix
  $archiveRoot = Join-Path $stage 'archive'
  [IO.Directory]::CreateDirectory($archiveRoot) | Out-Null
  Move-Item -LiteralPath $package -Destination (Join-Path $archiveRoot $name)
  [IO.Directory]::CreateDirectory($out) | Out-Null
  $archive = Join-Path $out ($name + '.zip')
  if (Test-Path -LiteralPath $archive) { throw "An immutable package already exists: $archive" }
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  [IO.Compression.ZipFile]::CreateFromDirectory($archiveRoot, $archive, [IO.Compression.CompressionLevel]::Optimal, $false)
  $listing = [ordered]@{schema='inferssd-package 1'; name=$name; platform='windows-x64'; backend=$Backend; minimum_windows='10.0.22000'; minimum_cpu='avx2'; file=($name+'.zip'); size=(Get-Item -LiteralPath $archive).Length; sha256=(Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()}
  if ($Backend -eq 'cuda') { $listing['minimum_nvidia_driver'] = $MinimumNvidiaDriver }
  foreach ($key in 'version', 'build', 'revision', 'date', 'branch', 'label') { $listing[$key] = $version.$key }
  [IO.File]::WriteAllText((Join-Path $out ($name+'.json')), ($listing | ConvertTo-Json -Depth 5), $utf8)
  Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'install.ps1') -Destination (Join-Path $out 'install.ps1')
  Write-Output "Packaged $archive"
} finally {
  $absolute = [IO.Path]::GetFullPath($stage)
  if ([IO.Path]::GetDirectoryName($absolute) -ne $temporary -or ![IO.Path]::GetFileName($absolute).StartsWith('ip-')) { throw 'Refusing cleanup outside the temporary staging directory.' }
  if (Test-Path -LiteralPath $absolute) { Remove-Item -LiteralPath $absolute -Recurse -Force }
}
