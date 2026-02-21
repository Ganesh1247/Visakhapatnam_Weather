param(
    [Parameter(Mandatory = $true)]
    [string]$SpaceId,

    [Parameter(Mandatory = $true)]
    [string]$HFToken
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git is required but not found in PATH."
}

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

git lfs install | Out-Host

$remoteUrl = "https://user:$HFToken@huggingface.co/spaces/$SpaceId"
git push $remoteUrl HEAD:main

Write-Host ""
Write-Host "Deployed to: https://huggingface.co/spaces/$SpaceId"
