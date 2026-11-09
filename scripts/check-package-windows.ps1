param([Parameter(Mandatory=$true)][string]$Archive)
# Install and exercise the relocated archive with no Git, Python, compiler, SDK, CMake or Ninja on PATH.
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archivePath = (Resolve-Path -LiteralPath $Archive).Path
$manifest = Get-Content -LiteralPath ([IO.Path]::ChangeExtension($archivePath, '.json')) -Raw | ConvertFrom-Json
if ($manifest.schema -ne 'inferssd-package 1' -or $manifest.platform -ne 'windows-x64' -or
    $manifest.size -ne (Get-Item -LiteralPath $archivePath).Length -or
    $manifest.sha256 -ne (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash) { throw 'Invalid package manifest or bytes.' }
$temporary = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\')
$work = Join-Path $temporary ('iw-check-' + [Guid]::NewGuid().ToString('N').Substring(0,12))
[IO.Directory]::CreateDirectory($work) | Out-Null
$previousPath = $env:PATH
$names = 'INFERSSD_HOME', 'INFERSSD_BIN_DIR', 'INFERSSD_PACKAGES', 'INFERSSD_RESOURCES', 'INFERSSD_REF', 'INFERSSD_FROM_SOURCE', 'PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV'
$previous = @{}
foreach ($key in $names) { $previous[$key] = [Environment]::GetEnvironmentVariable($key, 'Process') }
try {
  $place = Join-Path $work ('unpacked here ' + [char]0x4F60 + [char]0x597D)
  [IO.Compression.ZipFile]::ExtractToDirectory($archivePath, $place)
  $package = Join-Path $place $manifest.name
  if (!(Test-Path -LiteralPath (Join-Path $package 'bin\inferssd.exe'))) { throw 'Package entry point is missing.' }
  $tools = Join-Path $work 'tools'
  [IO.Directory]::CreateDirectory($tools) | Out-Null
  foreach ($tool in 'git', 'python', 'python3', 'pip', 'cl', 'clang', 'clang++', 'cmake', 'ninja', 'msbuild') {
    [IO.File]::WriteAllText((Join-Path $tools ($tool+'.cmd')), "@echo $tool called>>`"$work\developer-tools-called`"`r`n@exit /b 127`r`n")
  }
  $env:PATH = "$tools;$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\WindowsPowerShell\v1.0"
  foreach ($key in $names) { [Environment]::SetEnvironmentVariable($key, $null, 'Process') }
  $env:INFERSSD_HOME = Join-Path $work 'user data'
  $env:INFERSSD_BIN_DIR = Join-Path $work 'user bin'
  $env:INFERSSD_RESOURCES = Join-Path $package 'engine-resources.json'
  $python = Join-Path $package 'python\python.exe'
  & $python -I -X utf8 (Join-Path $PSScriptRoot 'check-package-windows.py') $package $work
  if ($LASTEXITCODE -ne 0) { throw 'Relocated package checks failed.' }
  $listing = $manifest | Select-Object *
  $listing | Add-Member -Force -NotePropertyName url -NotePropertyValue ([Uri]$archivePath).AbsoluteUri
  $listPath = Join-Path $work 'latest-windows.json'
  [IO.File]::WriteAllText($listPath, ($listing | ConvertTo-Json -Depth 8), [Text.UTF8Encoding]::new($false))
  $env:INFERSSD_PACKAGES = ([Uri]$listPath).AbsoluteUri
  [Environment]::SetEnvironmentVariable('INFERSSD_RESOURCES', $null, 'Process')
  & (Join-Path $PSScriptRoot 'install.ps1')
  if ($LASTEXITCODE -ne 0) { throw 'Installing the package failed.' }
  $installed = Join-Path $env:INFERSSD_HOME 'current\bin\inferssd.exe'
  $version = (& $installed --version --json | ConvertFrom-Json)
  if ($LASTEXITCODE -ne 0 -or $version.label -ne $manifest.label) { throw 'Installed package version mismatch.' }
  & $installed serve --help | Out-Null
  if ($LASTEXITCODE -ne 0) { throw 'Installed command failed.' }
  if (Test-Path -LiteralPath (Join-Path $work 'developer-tools-called')) { throw 'Installation or runtime called a developer tool.' }
  Write-Output 'Precompiled package relocated, installed and ran without developer tools.'
} finally {
  $env:PATH = $previousPath
  foreach ($key in $names) { [Environment]::SetEnvironmentVariable($key, $previous[$key], 'Process') }
  $absolute = [IO.Path]::GetFullPath($work)
  if ([IO.Path]::GetDirectoryName($absolute) -ne $temporary -or ![IO.Path]::GetFileName($absolute).StartsWith('iw-check-')) { throw 'Refusing cleanup outside test staging directory.' }
  if (Test-Path -LiteralPath $absolute) { Remove-Item -LiteralPath $absolute -Recurse -Force }
}
