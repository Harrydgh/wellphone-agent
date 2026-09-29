param(
    [Parameter(Mandatory = $true)]
    [string]$Task,
    [string]$Model = $(if ($env:WELLPHONE_DEEPSEEK_MODEL) { $env:WELLPHONE_DEEPSEEK_MODEL } else { "deepseek-flash" }),
    [switch]$Concurrent,
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"
$projectPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $projectPython)) {
    throw "Project environment is missing. Run .\scripts\setup.ps1 first."
}

$arguments = @(
    "-m", "wellphone_agent", "shopping-agent",
    "--task", $Task,
    "--model", $Model
)
if ($Concurrent) {
    $arguments += "--concurrent"
}
if ($PlanOnly) {
    $arguments += "--dry-run"
}

Push-Location $projectRoot
try {
    & $projectPython @arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
