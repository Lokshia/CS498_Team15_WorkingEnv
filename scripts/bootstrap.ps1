[CmdletBinding()]
param(
    [ValidateSet('cu130', 'cpu')]
    [string]$Backend = 'cu130'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonVersion = '3.13.14'
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'Install uv >= 0.11.21 first: https://docs.astral.sh/uv/getting-started/installation/'
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw 'Git must be on PATH to fetch the pinned course and mjlab dependencies.'
}
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'uv.lock'))) {
    throw 'uv.lock is missing. Obtain the complete team release; do not resolve new versions locally.'
}
if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw 'Use scripts/bootstrap.sh on Linux. This release targets Windows/Linux x86-64.'
}

$hadProjectEnvironment = Test-Path Env:UV_PROJECT_ENVIRONMENT
$previousProjectEnvironment = [Environment]::GetEnvironmentVariable('UV_PROJECT_ENVIRONMENT', 'Process')
$hadPythonDownloads = Test-Path Env:UV_PYTHON_DOWNLOADS
$previousPythonDownloads = [Environment]::GetEnvironmentVariable('UV_PYTHON_DOWNLOADS', 'Process')
Push-Location -LiteralPath $projectRoot
try {
    # An inherited environment override must not install into another project.
    $env:UV_PROJECT_ENVIRONMENT = Join-Path $projectRoot '.venv'
    # Sync reuses an exact interpreter or downloads it when absent. A separate
    # python install can unnecessarily rewrite existing global Python links.
    $env:UV_PYTHON_DOWNLOADS = 'automatic'
    & uv sync --locked --no-dev --python $pythonVersion --extra $Backend
    if ($LASTEXITCODE -ne 0) { throw 'Locked dependency installation failed.' }
    $pythonExecutable = Join-Path $projectRoot '.venv\Scripts\python.exe'
    $device = if ($Backend -eq 'cu130') { 'cuda' } else { 'cpu' }
    & $pythonExecutable -m lifting_lab doctor --device $device
    if ($LASTEXITCODE -ne 0) {
        throw "The $device doctor check failed. This setup did not fall back to another device."
    }
    Write-Host "Environment ready for $device. Use this interpreter: $pythonExecutable"
    Write-Host 'Do not run bootstrap or dependency updates while a training job is active.'
}
finally {
    if ($hadProjectEnvironment) {
        [Environment]::SetEnvironmentVariable('UV_PROJECT_ENVIRONMENT', $previousProjectEnvironment, 'Process')
    } else {
        Remove-Item Env:UV_PROJECT_ENVIRONMENT -ErrorAction SilentlyContinue
    }
    if ($hadPythonDownloads) {
        [Environment]::SetEnvironmentVariable('UV_PYTHON_DOWNLOADS', $previousPythonDownloads, 'Process')
    } else {
        Remove-Item Env:UV_PYTHON_DOWNLOADS -ErrorAction SilentlyContinue
    }
    Pop-Location
}
