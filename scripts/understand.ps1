param(
    [string]$App = "com.android.settings"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"
$projectPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $projectPython)) {
    throw "Project environment is missing. Run .\scripts\setup.ps1 first."
}

Push-Location $projectRoot
try {
    & $projectPython -m wellphone_agent understand --app $App
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
