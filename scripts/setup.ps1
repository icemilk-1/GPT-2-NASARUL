# GPT4RUL one-click environment setup (Windows)
# Usage: .\scripts\setup.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host "=== GPT4RUL Setup ===" -ForegroundColor Cyan
Write-Host "Project root: $Root"

# Python check
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
    Write-Host "ERROR: python not found. Install Python 3.8+ first." -ForegroundColor Red
    exit 1
}
python --version

# Virtual environment
$venv = Join-Path $Root ".venv"
$pip = Join-Path $venv "Scripts\pip.exe"
$python = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Host "Creating virtual environment ..."
    python -m venv $venv
}

Write-Host "Installing dependencies ..."
& $pip install -r requirements.txt -q

Write-Host "Downloading GPT-2 (first run only) ..."
& $python scripts/download_gpt2.py

Write-Host "Checking C-MAPSS data ..."
& $python scripts/check_data.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Setup paused: please download C-MAPSS data (see data/README.md), then re-run setup." -ForegroundColor Yellow
    exit 1
}

Write-Host "`n=== Setup complete ===" -ForegroundColor Green
Write-Host "Next: .\scripts\reproduce.ps1"
