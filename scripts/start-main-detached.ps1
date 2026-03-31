Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$runDir = Join-Path $repoRoot ".run"
$backendDir = Join-Path $repoRoot "backend"
$frontendDir = Join-Path $repoRoot "frontend"

New-Item -ItemType Directory -Path $runDir -Force | Out-Null

function Stop-ExistingRepoProcess {
    param(
        [string]$MatchText
    )

    $procs = Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -and $_.CommandLine -like "*$MatchText*" }
    foreach ($proc in $procs) {
        try {
            Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
        }
        catch {
        }
    }
}

function Get-ListeningProcessId {
    param(
        [int]$Port
    )

    $conn = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -eq $conn) {
        return $null
    }
    return $conn.OwningProcess
}

function Stop-RepoPortListener {
    param(
        [int]$Port
    )

    $owningProcess = Get-ListeningProcessId -Port $Port
    if (-not $owningProcess) {
        return
    }

    $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $owningProcess" -ErrorAction SilentlyContinue
    if ($proc -and $proc.CommandLine -and $proc.CommandLine -like "*$repoRoot*") {
        try {
            Stop-Process -Id $owningProcess -Force -ErrorAction Stop
        }
        catch {
        }
    }
}

Stop-ExistingRepoProcess -MatchText "vector-database\\backend"
Stop-ExistingRepoProcess -MatchText "vector-database\\frontend"
Stop-RepoPortListener -Port 8000
Stop-RepoPortListener -Port 5173

Start-Sleep -Seconds 1

if (Get-ListeningProcessId -Port 8000) {
    throw "Port 8000 is already in use by a non-repo process."
}
if (Get-ListeningProcessId -Port 5173) {
    throw "Port 5173 is already in use by a non-repo process."
}

$backendCommand = (
    'set "VECTOR_DB_DEVICE=cpu" && ' +
    'set "VECTOR_DB_CPU_THREADS=2" && ' +
    'set "INDEX_THROTTLE_MS=15" && ' +
    'set "INDEX_BATCH_COOLDOWN_MS=150" && ' +
    'set "OMP_NUM_THREADS=2" && ' +
    'set "MKL_NUM_THREADS=2" && ' +
    '"{0}\venv\Scripts\python.exe" -m uvicorn server:app --host 127.0.0.1 --port 8000'
) -f $repoRoot

$backendProc = Start-Process `
    -FilePath "cmd.exe" `
    -ArgumentList @("/c", $backendCommand) `
    -WorkingDirectory $backendDir `
    -RedirectStandardOutput (Join-Path $runDir "backend.out.log") `
    -RedirectStandardError (Join-Path $runDir "backend.err.log") `
    -PassThru

$frontendProc = Start-Process `
    -FilePath "npm.cmd" `
    -ArgumentList @("run", "dev", "--", "--host", "127.0.0.1", "--port", "5173") `
    -WorkingDirectory $frontendDir `
    -RedirectStandardOutput (Join-Path $runDir "frontend.out.log") `
    -RedirectStandardError (Join-Path $runDir "frontend.err.log") `
    -PassThru

Set-Content -Path (Join-Path $runDir "backend.pid") -Value $backendProc.Id -Encoding ascii
Set-Content -Path (Join-Path $runDir "frontend.pid") -Value $frontendProc.Id -Encoding ascii

for ($attempt = 0; $attempt -lt 45; $attempt++) {
    if (Get-ListeningProcessId -Port 8000) {
        break
    }
    Start-Sleep -Seconds 1
}

for ($attempt = 0; $attempt -lt 15; $attempt++) {
    if (Get-ListeningProcessId -Port 5173) {
        break
    }
    Start-Sleep -Seconds 1
}

if (-not (Get-ListeningProcessId -Port 8000)) {
    throw "Backend did not bind to port 8000. Check .run\\backend.err.log"
}
if (-not (Get-ListeningProcessId -Port 5173)) {
    throw "Frontend did not bind to port 5173. Check .run\\frontend.err.log"
}

Write-Output "Backend PID: $($backendProc.Id)"
Write-Output "Frontend PID: $($frontendProc.Id)"
Write-Output "Frontend URL: http://127.0.0.1:5173/"