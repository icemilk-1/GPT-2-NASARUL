# Push to GitHub after creating an empty repo named GPT4RUL-Reproduction
# Usage:
#   1. Create empty repo at https://github.com/new  (no README/license)
#   2. .\scripts\push_github.ps1 -Username YOUR_GITHUB_USERNAME

param(
    [Parameter(Mandatory = $true)]
    [string]$Username,
    [string]$RepoName = "GPT4RUL-Reproduction"
)

$ErrorActionPreference = "Stop"
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$remote = "https://github.com/$Username/$RepoName.git"
Write-Host "Remote: $remote"

$existing = git remote get-url origin 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "Updating existing origin: $existing -> $remote"
    git remote set-url origin $remote
} else {
    git remote add origin $remote
}

git branch -M main
git push -u origin main

Write-Host "`nDone! Repo URL: https://github.com/$Username/$RepoName"
