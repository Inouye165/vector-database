Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$runDir = Join-Path $repoRoot ".run"

function Stop-PidFileProcess {
    param(
        [string]$PidFile
    )

    if (-not (Test-Path $PidFile)) {
        return
    }

    $pidValue = Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($pidValue) {
        try {
            Stop-Process -Id ([int]$pidValue) -Force -ErrorAction Stop
        }
        catch {
        }
    }

    Remove-Item $PidFile -ErrorAction SilentlyContinue
}

function Stop-RepoProcess {
    param(
        [string]$MatchText
    )

    Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -and $_.CommandLine -like "*$MatchText*" } |
        ForEach-Object {
            try {
                Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop
            }
            catch {
            }
        }
}

function Stop-RepoPortListener {
    param(
        [int]$Port
    )

    $owningProcess = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty OwningProcess
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

Stop-PidFileProcess -PidFile (Join-Path $runDir "backend.pid")
Stop-PidFileProcess -PidFile (Join-Path $runDir "frontend.pid")
Stop-RepoProcess -MatchText "vector-database\\backend"
Stop-RepoProcess -MatchText "vector-database\\frontend"
Stop-RepoPortListener -Port 8000
Stop-RepoPortListener -Port 5173

Write-Output "Detached main app processes stopped."