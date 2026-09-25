param(
    [double]$Duration = 10,
    [string]$App = "com.android.browser"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"

Push-Location $projectRoot
try {
    python -m wellphone_agent display-test --duration $Duration --app $App
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}

