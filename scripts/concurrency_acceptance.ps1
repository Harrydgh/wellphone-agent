param(
    [Parameter(Mandatory = $true)]
    [string]$Query,
    [string]$Model = $(if ($env:WELLPHONE_DEEPSEEK_MODEL) { $env:WELLPHONE_DEEPSEEK_MODEL } else { "deepseek-flash" })
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
    & $projectPython -m wellphone_agent meituan-agent --query $Query --model $Model --concurrent
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
