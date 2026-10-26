[CmdletBinding()]
param(
    [ValidateSet('cuda', 'cpu')]
    [string]$Device = 'cuda',
    [int]$NumEnvs = 512,
    [int]$Iterations = 500,
    [int]$Seed = 11,
    [string]$Root = '',
    [switch]$Quick
)

# Three forearm-support training runs with fixed folder names, each followed
# by the team evaluation protocol, so everyone can see which checkpoint to
# share. 1_stand_hold is the diagnostic baseline (a 1 g box: standing and
# keeping the box on the forearms are the same episode); 2_curriculum and
# 3_mass_range are the two ways of learning several weights. Nothing here
# uses pushes or payload randomization. -Quick only checks the plumbing.

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Run scripts\bootstrap.ps1 first; expected $python"
}
if ($Root -eq '') {
    $Root = Join-Path $projectRoot ('outputs\milestones_' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
}
New-Item -ItemType Directory -Path $Root | Out-Null
$evalEnvs = if ($Quick) { [Math]::Min($NumEnvs, 32) } else { 32 }
$common = @('--device', $Device, '--support-mode', 'forearms', '--seed', "$Seed")

function Invoke-Lab([string]$Label, [string[]]$Arguments) {
    Write-Host "=== $Label"
    $log = Join-Path $Root ($Label + '.log')
    # Warp and MuJoCo print warnings on stderr; under 'Stop' PowerShell 5.1
    # would turn the first such line into a terminating error. Stringify the
    # merged stream and rely on the exit code instead.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $python -u -m lifting_lab @Arguments 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log |
            Select-String -NotMatch 'Module .* load on device|warnings.warn|UserWarning|NativeCommandError|^\s*$'
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($LASTEXITCODE -ne 0) { throw "$Label failed; see $log" }
}

$runs = @(
    @{ Name = '1_stand_hold';  Train = @('--mass', '0');                        Checkpoint = 'policy.pt' },
    @{ Name = '2_curriculum';  Train = @('--curriculum');                       Checkpoint = 'phase_2\policy.pt' },
    @{ Name = '3_mass_range';  Train = @('--mass-range-fraction', '0', '0.05'); Checkpoint = 'policy.pt' }
)
foreach ($run in $runs) {
    $outDir = Join-Path $Root $run.Name
    Invoke-Lab "train_$($run.Name)" (@('train') + $common + @('--num-envs', "$NumEnvs", '--iterations', "$Iterations", '--output', $outDir) + $run.Train)
    $evalArgs = @('evaluate') + $common + @('--num-envs', "$evalEnvs", '--checkpoint', (Join-Path $outDir $run.Checkpoint), '--output', (Join-Path $Root "eval_$($run.Name)"))
    if ($Quick) { $evalArgs += '--quick' }
    Invoke-Lab "eval_$($run.Name)" $evalArgs
}

Write-Host ''
Write-Host ("{0,-14} {1,8} {2,8} {3,8} {4,8}   checkpoint to share (with experiment.json + completed.json)" -f 'run', '1.25%', '3.75%', '7.5%', 'overall')
foreach ($run in $runs) {
    $result = Get-Content (Join-Path $Root "eval_$($run.Name)\evaluation.json") -Raw | ConvertFrom-Json
    $rates = $result.condition_summaries | ForEach-Object { '{0,7:P0}' -f $_.success_rate }
    Write-Host ("{0,-14} {1} {2} {3} {4,8:P0}   {5}" -f $run.Name, $rates[0], $rates[1], $rates[2], $result.success_rate, (Join-Path $Root (Join-Path $run.Name $run.Checkpoint)))
}
if ($Quick) {
    Write-Host 'Quick mode: 2 s episodes, plumbing only. Re-run without -Quick for protocol numbers.'
}
Write-Host "All outputs: $Root"
