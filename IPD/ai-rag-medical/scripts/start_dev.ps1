[CmdletBinding()]
param(
    [int]$ApiPort = 8010,
    [int]$WebPort = 5173,
    [string]$PythonExe = "e:/Coas/.venv/Scripts/python.exe",
    [switch]$BackendOnly
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$frontendRoot = Join-Path $projectRoot "frontend"

function Stop-PortListener {
    param([int]$Port)

    $listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if (-not $listeners) {
        Write-Host "[info] Port $Port is free."
        return
    }

    $processIds = $listeners | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($processId in $processIds) {
        try {
            Stop-Process -Id $processId -Force -ErrorAction Stop
            Write-Host "[ok] Stopped PID $processId on port $Port"
        }
        catch {
            Write-Host "[warn] Could not stop PID $processId on port $Port"
        }
    }
}

function Resolve-PythonCommand {
    param([string]$Preferred)

    if (Test-Path $Preferred) {
        return $Preferred
    }

    $pyCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pyCmd) {
        return "python"
    }

    throw "Python executable not found. Use -PythonExe <path>."
}

function Resolve-NpmCommand {
    $viteCmd = Join-Path $frontendRoot "node_modules\.bin\vite.cmd"
    if (Test-Path $viteCmd) {
        return $viteCmd
    }

    throw "Vite executable not found. Run npm install in the frontend folder first."
}

if (-not (Test-Path $frontendRoot) -and -not $BackendOnly) {
    throw "Frontend folder not found: $frontendRoot"
}

$pythonCmd = Resolve-PythonCommand -Preferred $PythonExe
$viteCmd = Resolve-NpmCommand

Stop-PortListener -Port $ApiPort
if (-not $BackendOnly) {
    Stop-PortListener -Port $WebPort
}

Start-Process -FilePath $pythonCmd -WorkingDirectory $projectRoot -ArgumentList @("scripts/run_api.py", "--host", "127.0.0.1", "--port", $ApiPort) | Out-Null
Write-Host "[ok] Backend starting at http://127.0.0.1:$ApiPort"

if (-not $BackendOnly) {
    Start-Process -FilePath $viteCmd -WorkingDirectory $frontendRoot -ArgumentList @("--host", "127.0.0.1", "--port", $WebPort) | Out-Null
    Write-Host "[ok] Frontend starting at http://127.0.0.1:$WebPort"
}

Write-Host "[done] Startup commands dispatched."