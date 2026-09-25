$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $projectRoot ".venv"
$python = Join-Path $venv "Scripts\python.exe"

Push-Location $projectRoot
try {
    if (-not (Test-Path $python)) {
        python -m venv $venv
    }
    & $python -m pip install --upgrade pip
    & $python -m pip install -e .
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
