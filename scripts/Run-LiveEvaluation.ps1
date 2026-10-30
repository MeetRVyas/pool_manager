[CmdletBinding()]
param(
    [ValidateSet('diagnostic-paced-100', 'diagnostic-burst-100', 'diagnostic-12-12rpm', 'soak-1000-60', 'lite35-1000-30rpm', 'lite35-100-24rpm', 'lites-1900-48rpm', 'lites-1400-36rpm', 'lites-1900-60rpm', 'lites-2000-60rpm', 'lite-100-30', 'lite-100-60', 'lite-burst-100', 'flash-12-60')]
    [string]$TestProfile = 'soak-1000-60',
    [string]$RunLabel = 'fresh',
    [string]$EnvFile = '.env',
    [switch]$Run,
    [switch]$Diagnostics,
    [switch]$DiagnosticStackDumps
)

$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw "Project Python missing. Create .venv and install requirements.txt first; see evals\README.md."
}
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
Push-Location -LiteralPath $taskRoot
try {
    $taskArgs = @('-u', '-m', 'evals.run_live_profile', '--profile', $TestProfile, '--env-file', $EnvFile, '--label', $RunLabel)
    if ($Diagnostics) { $taskArgs += '--diagnostics' }
    if ($DiagnosticStackDumps) { $taskArgs += '--diagnostic-stack-dumps' }
    if ($Run) { $taskArgs += '--yes' } else { $taskArgs += '--prepare' }
    & $taskPython @taskArgs
    $taskExit = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $taskExit
