param(
    [ValidateSet("WLAN")]
    [string]$Target = "WLAN"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"
$projectPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $projectPython)) { $projectPython = "python" }

Push-Location $projectRoot
try {
    & $projectPython -m wellphone_agent agent-test --target $Target
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
