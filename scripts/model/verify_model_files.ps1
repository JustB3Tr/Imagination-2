param(
    [string]$ModelDir = ""
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..\..")
if ([string]::IsNullOrWhiteSpace($ModelDir)) {
    $ModelDir = Join-Path $repoRoot "model"
}

if (!(Test-Path $ModelDir)) {
    throw "Model directory does not exist: $ModelDir"
}

$required = @(
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json"
)

$missing = @()
foreach ($name in $required) {
    $path = Join-Path $ModelDir $name
    if (!(Test-Path $path)) {
        $missing += $name
    }
}

$hasWeights = (Get-ChildItem -Path $ModelDir -Filter "*.safetensors" -File -ErrorAction SilentlyContinue | Measure-Object).Count -gt 0
$hasIndex = Test-Path (Join-Path $ModelDir "model.safetensors.index.json")
if (!($hasWeights -or $hasIndex)) {
    $missing += "model weights (*.safetensors or model.safetensors.index.json)"
}

if ($missing.Count -gt 0) {
    Write-Error ("Missing required model artifacts:`n- " + ($missing -join "`n- "))
    exit 1
}

Write-Host "[verify] model directory looks valid: $ModelDir"
Write-Host "[verify] required config/tokenizer files and weights are present."
