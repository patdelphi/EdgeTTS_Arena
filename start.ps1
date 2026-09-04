# One-click launcher for EdgeTTS-Arena (Windows PowerShell).
#
# It sources every dedicated worker env script that exists (Qwen3 / CosyVoice /
# MeloTTS), reports which worker interpreters are now configured, then starts the
# Arena. Models whose env script is missing are simply reported, not fatal.
#
# Usage (from the repo root):
#   .\start.ps1              # serve with the Gradio Arena UI (default)
#   .\start.ps1 -NoUi        # serve the JSON API only
#   .\start.ps1 -Doctor      # run the deployment doctor and exit
#   .\start.ps1 -Suite piper kokoro   # run the standard suite for given models
param(
    [switch]$NoUi,
    [switch]$Doctor,
    [string[]]$Suite,
    [string]$VenvPython = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

# 1) Source dedicated worker environments if they have been bootstrapped.
$envScripts = @(
    "exports\bootstrap\qwen3\env.ps1",
    "exports\bootstrap\cosyvoice\env.ps1",
    "exports\bootstrap\melotts\env.ps1"
)
foreach ($script in $envScripts) {
    $path = Join-Path $root $script
    if (Test-Path $path) {
        Write-Host "[start] sourcing $script" -ForegroundColor Cyan
        . $path
    }
    else {
        Write-Host "[start] skip $script (not bootstrapped yet)" -ForegroundColor DarkGray
    }
}

# 2) Report which dedicated worker interpreters are now visible to this shell.
$workers = @(
    @("EDGETTS_ARENA_QWEN3_PYTHON", "Qwen3-TTS 0.6B (PyTorch)"),
    @("EDGETTS_ARENA_COSYVOICE_PYTHON", "CosyVoice 300M SFT"),
    @("EDGETTS_ARENA_MELOTTS_PYTHON", "MeloTTS Chinese")
)
foreach ($pair in $workers) {
    $val = [Environment]::GetEnvironmentVariable($pair[0])
    if ($val) {
        Write-Host ("[start] {0} worker -> {1}" -f $pair[1], $val) -ForegroundColor Green
    }
    else {
        Write-Host ("[start] {0} worker NOT configured (that model will error until bootstrapped)" -f $pair[1]) -ForegroundColor Yellow
    }
}

# 3) Launch using the main venv interpreter.
if (-not (Test-Path $VenvPython)) {
    throw "Main venv Python not found: $VenvPython. Create it first (see README '本地部署测试')."
}

if ($Doctor) {
    & $VenvPython -m edgetts_arena doctor --workers --exports-root exports\doctor
    return
}

if ($Suite -and $Suite.Count -gt 0) {
    & $VenvPython -m edgetts_arena suite --models @Suite
    return
}

if ($NoUi) {
    Write-Host "[start] launching API: http://127.0.0.1:8000" -ForegroundColor Cyan
    & $VenvPython -m edgetts_arena serve
}
else {
    Write-Host "[start] launching Arena UI: http://127.0.0.1:8000/arena/" -ForegroundColor Cyan
    & $VenvPython -m edgetts_arena serve --ui
}
