$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "============================================================"
Write-Host "  Lingyin Lite v1.0.0"
Write-Host "============================================================"
Write-Host ""

$venvPy = Join-Path $PSScriptRoot "venv\Scripts\python.exe"

if (Test-Path $venvPy) {
    Write-Host "[INFO] Using project venv" -ForegroundColor Cyan
    & $venvPy run.py @args
} else {
    $sysPy = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $sysPy) {
        Write-Host "[ERROR] Python not found. Install Python 3.10+ and enable Add to PATH." -ForegroundColor Red
        Write-Host "https://www.python.org/downloads/"
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "[INFO] Using system Python to bootstrap venv" -ForegroundColor Cyan
    python run.py @args
}

Read-Host "Press Enter to exit"
