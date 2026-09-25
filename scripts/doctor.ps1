$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"

Push-Location $projectRoot
try {
    python -m wellphone_agent doctor
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
