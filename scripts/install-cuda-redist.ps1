# SPDX-License-Identifier: Apache-2.0
# Install the parts of NVIDIA's CUDA toolkit that a Windows build needs (compiler, runtime, CCCL and
# cuBLAS) from NVIDIA's redistributable archives, each checked against NVIDIA's manifest, into one
# directory. In GitHub Actions the directory becomes CUDA_PATH for the steps after this one.
param(
  [Parameter(Mandatory = $true)][string]$Version,
  [Parameter(Mandatory = $true)][string]$Root
)
$ErrorActionPreference = 'Stop'
$base = 'https://developer.download.nvidia.com/compute/cuda/redist/'
$manifest = Invoke-RestMethod "${base}redistrib_$Version.json"
$work = Join-Path ([IO.Path]::GetTempPath()) ('cuda-redist-' + [Guid]::NewGuid().ToString('N').Substring(0, 12))
New-Item -ItemType Directory -Force $Root, $work | Out-Null
try {
  foreach ($name in 'cuda_nvcc', 'cuda_cudart', 'cuda_cccl', 'libcublas', 'cuda_nvtx', 'cuda_profiler_api') {
    $part = $manifest.$name.'windows-x86_64'
    $zip = Join-Path $work (Split-Path $part.relative_path -Leaf)
    # The server can end a large response early: resume until the archive is whole.
    for ($i = 0; $i -lt 20 -and -not ((Test-Path $zip) -and (Get-Item $zip).Length -ge [int64]$part.size); $i++) {
      curl.exe -sSL -C - -o $zip ($base + $part.relative_path)
    }
    if ((Get-FileHash -Algorithm SHA256 $zip).Hash -ne $part.sha256) { throw "$name does not match the manifest" }
    tar -xf $zip -C $work
    if ($LASTEXITCODE) { throw "$name does not extract" }
    $extracted = Join-Path $work ([IO.Path]::GetFileNameWithoutExtension($zip))
    Copy-Item -Recurse -Force (Join-Path $extracted '*') $Root
  }
} finally {
  Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
}
if ($env:GITHUB_ENV) { "CUDA_PATH=$Root" >> $env:GITHUB_ENV }
