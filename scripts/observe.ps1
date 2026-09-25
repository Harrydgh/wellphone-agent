param(
    [string]$Url = "https://www.baidu.com/s?wd=Android",
    [string]$App = "com.android.browser"
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
    & $projectPython -m wellphone_agent observe --url $Url --app $App
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
