Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$testerDir = $PSScriptRoot
$repoRoot = Split-Path -Parent (Split-Path -Parent $testerDir)
$toolVenvPython = Join-Path $testerDir ".venv\Scripts\python.exe"
$seedPython = Join-Path $repoRoot "venv\Scripts\python.exe"
$requirementsPath = Join-Path $testerDir "requirements.txt"

if (-not (Test-Path $toolVenvPython)) {
    if (-not (Test-Path $seedPython)) {
        throw "Missing seed interpreter at $seedPython"
    }

    & $seedPython -m venv (Join-Path $testerDir ".venv")
}

$needsInstall = $false
try {
    & $toolVenvPython -c "import ultralytics, torch, torchvision, PIL"
}
catch {
    $needsInstall = $true
}

if ($needsInstall) {
    & $toolVenvPython -m pip install -r $requirementsPath
}

Push-Location $repoRoot
try {
    & $toolVenvPython -m tools.bbox_crop_tester.app
}
finally {
    Pop-Location
}