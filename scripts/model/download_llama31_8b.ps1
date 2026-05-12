param(
    [string]$RepoId = "meta-llama/Llama-3.1-8B-Instruct",
    [string]$TargetDir = "",
    [string]$Revision = "main"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..\..")
if ([string]::IsNullOrWhiteSpace($TargetDir)) {
    $TargetDir = Join-Path $repoRoot "model"
}

New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null

$pyScript = Join-Path $scriptDir "download_llama31_8b.py"
if (!(Test-Path $pyScript)) {
    throw "Missing python downloader: $pyScript"
}

python $pyScript --repo-id $RepoId --target-dir $TargetDir --revision $Revision
if ($LASTEXITCODE -ne 0) {
    throw "Download failed with exit code $LASTEXITCODE"
}

Write-Host "[download] done."
