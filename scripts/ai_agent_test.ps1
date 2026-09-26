param(
    [string]$Task = "打开WLAN设置",
    [string]$Model = $(if ($env:WELLPHONE_OPENAI_MODEL) { $env:WELLPHONE_OPENAI_MODEL } else { "gpt-6-astra" })
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"
$projectPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $projectPython)) { $projectPython = "python" }

Push-Location $projectRoot
try {
    & $projectPython -m wellphone_agent ai-agent --task $Task --model $Model
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
