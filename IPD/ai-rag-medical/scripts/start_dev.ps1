<#
  Menjalankan stack development Medical RAG dalam satu perintah:

  1. (Opsional) Rebuild indeks materi — -RebuildIndex
  2. API FastAPI (scripts/run_api.py)
  3. Dev server Vite (frontend), kecuali -BackendOnly
  4. Membuka browser ke UI web, kecuali -NoBrowser

  Contoh:
    powershell -ExecutionPolicy Bypass -File scripts/start_dev.ps1
    powershell -ExecutionPolicy Bypass -File scripts/start_dev.ps1 -RebuildIndex
    powershell -ExecutionPolicy Bypass -File scripts/start_dev.ps1 -BackendOnly
#>
[CmdletBinding()]
param(
    [int]$ApiPort = 8010,
    [int]$WebPort = 5173,
    [string]$PythonExe = "e:/Coas/.venv/Scripts/python.exe",
    [switch]$BackendOnly,
    [switch]$NoBrowser,
    [switch]$RebuildIndex
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

if ($RebuildIndex) {
    $buildScript = Join-Path $projectRoot "scripts\build_index.py"
    Write-Host "[info] Rebuilding index (Chroma vector skipped for speed)..."
    & $pythonCmd $buildScript @("--skip-vector")
    if (-not $?) {
        throw "build_index failed."
    }
    Write-Host "[ok] Index build finished."
}

$viteCmd = $null
if (-not $BackendOnly) {
    $viteCmd = Resolve-NpmCommand
}

Stop-PortListener -Port $ApiPort
if (-not $BackendOnly) {
    Stop-PortListener -Port $WebPort
}

Start-Process -FilePath $pythonCmd -WorkingDirectory $projectRoot -ArgumentList @("scripts/run_api.py", "--host", "127.0.0.1", "--port", $ApiPort) | Out-Null
Write-Host "[ok] Backend starting at http://127.0.0.1:$ApiPort (docs: /docs)"

if (-not $BackendOnly) {
    Start-Process -FilePath $viteCmd -WorkingDirectory $frontendRoot -ArgumentList @("--host", "127.0.0.1", "--port", $WebPort) | Out-Null
    Write-Host "[ok] Frontend starting at http://127.0.0.1:$WebPort"

    if (-not $NoBrowser) {
        $webUrl = "http://127.0.0.1:$WebPort"
        Start-Sleep -Seconds 2
        Start-Process $webUrl
        Write-Host "[ok] Opening browser: $webUrl"
    }
}

Write-Host "[done] Startup commands dispatched."
