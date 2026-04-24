<#
  Publikasikan data Medical RAG lokal ke Supabase + Cloudflare R2.

  Alur:
  1. Migrasi chunks/images/graph_edges dari SQLite lokal ke Supabase
  2. Upload file image lokal ke Cloudflare R2 lalu update storage_url di Supabase

  Contoh:
    powershell -ExecutionPolicy Bypass -File scripts/publish_cloud.ps1
#>
[CmdletBinding()]
param(
    [string]$PythonExe = "e:/Coas/.venv/Scripts/python.exe",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$workerWranglerPath = Join-Path $projectRoot "worker\wrangler.toml"

function Get-R2PublicBaseUrl {
    param([string]$Fallback = "https://pub-56e3e682628340078755345d5a0d6c05.r2.dev")

    if ($env:R2_PUBLIC_BASE_URL) {
        return $env:R2_PUBLIC_BASE_URL.Trim().Trim('"')
    }

    if (Test-Path $workerWranglerPath) {
        $match = Select-String -Path $workerWranglerPath -Pattern '^\s*R2_PUBLIC_BASE_URL\s*=\s*"([^"]+)"\s*$' | Select-Object -First 1
        if ($match -and $match.Matches.Count -gt 0) {
            return $match.Matches[0].Groups[1].Value.Trim()
        }
    }

    return $Fallback
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

function Invoke-PythonScript {
    param(
        [string]$PythonCommand,
        [string]$ScriptPath,
        [string[]]$Arguments = @()
    )

    if (-not (Test-Path $ScriptPath)) {
        throw "Script not found: $ScriptPath"
    }

    & $PythonCommand $ScriptPath @Arguments
    if (-not $?) {
        throw "Script failed: $ScriptPath"
    }
}

$pythonCmd = Resolve-PythonCommand -Preferred $PythonExe
$migrateScript = Join-Path $projectRoot "migration\02_migrate_to_supabase.py"
$uploadImagesScript = Join-Path $projectRoot "migration\05_upload_images_to_r2.py"
$r2PublicBaseUrl = Get-R2PublicBaseUrl
$env:R2_PUBLIC_BASE_URL = $r2PublicBaseUrl

if ($DryRun) {
    Write-Host "[dry-run] Would run: $pythonCmd $migrateScript"
    Write-Host "[dry-run] Would run: $pythonCmd $uploadImagesScript"
    Write-Host "[dry-run] R2_PUBLIC_BASE_URL = $r2PublicBaseUrl"
    return
}

Write-Host "[info] Migrating local index to Supabase..."
Invoke-PythonScript -PythonCommand $pythonCmd -ScriptPath $migrateScript

Write-Host "[info] Uploading images to Cloudflare R2..."
Invoke-PythonScript -PythonCommand $pythonCmd -ScriptPath $uploadImagesScript

Write-Host "[ok] Cloud publish finished."