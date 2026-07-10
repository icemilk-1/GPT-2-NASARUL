# GPT4RUL one-click full reproduction (Windows)
# Usage:
#   .\scripts\reproduce.ps1              # all 4 datasets (default)
#   .\scripts\reproduce.ps1 -DatasetId FD001
#   .\scripts\reproduce.ps1 -SkipSetup

param(
    [string]$DatasetId = "",
    [switch]$SkipSetup
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

$logDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir ("reproduce_{0:yyyyMMdd_HHmmss}.log" -f (Get-Date))

function Log($msg) {
    $line = "[$(Get-Date -Format 'HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $logFile -Value $line
}

Log "=== GPT4RUL Reproduction ==="
Log "Log file: $logFile"
Log "NOTE: Full 4-dataset run on CPU may take several hours."

if (-not $SkipSetup) {
    Log "Running setup ..."
    & "$Root\scripts\setup.ps1" 2>&1 | Tee-Object -FilePath $logFile -Append
}

Log "Checking data ..."
& $python scripts/check_data.py 2>&1 | Tee-Object -FilePath $logFile -Append
if ($LASTEXITCODE -ne 0) { exit 1 }

if ($DatasetId) {
    Log "Preprocessing $DatasetId ..."
    & $python src/preprocess_gpt4rul.py --dataset-id $DatasetId 2>&1 | Tee-Object -FilePath $logFile -Append
    Log "Training $DatasetId ..."
    & $python src/train_gpt4rul.py --dataset-id $DatasetId 2>&1 | Tee-Object -FilePath $logFile -Append
    Log "Evaluating $DatasetId ..."
    & $python src/evaluate_gpt4rul.py --dataset-id $DatasetId 2>&1 | Tee-Object -FilePath $logFile -Append
} else {
    Log "Preprocessing all datasets ..."
    & $python src/preprocess_gpt4rul.py --all 2>&1 | Tee-Object -FilePath $logFile -Append
    Log "Training all datasets ..."
    & $python src/train_gpt4rul.py --all 2>&1 | Tee-Object -FilePath $logFile -Append
    Log "Evaluating all datasets ..."
    & $python scripts/evaluate_all.py 2>&1 | Tee-Object -FilePath $logFile -Append
}

Log "=== Done ==="
Log "Results: results/gpt4rul_summary.csv"
Log "Compare with: results/expected/gpt4rul_summary.csv"
