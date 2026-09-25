param(
    [string]$Url = "https://example.com",
    [double]$Hold = 5,
    [string]$App = "com.android.browser"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"

Push-Location $projectRoot
try {
    python -m wellphone_agent demo --url $Url --hold $Hold --app $App
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}

