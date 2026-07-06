<#
.SYNOPSIS
    Build the wc_xpu Python wheel using uv + maturin.

.DESCRIPTION
    Creates (or reuses) a uv-managed virtual environment in .venv,
    installs the build backend (maturin), and builds a release wheel
    into the dist/ folder.

    wc_xpu is a native PyO3 extension for Intel XPU (OpenCL D3D11 interop),
    so the produced wheel is platform-specific (x64 Windows). It targets the
    abi3-py310 stable ABI, so a single wheel works for CPython 3.10+.

.PARAMETER Python
    Python version to use for the build environment. Default: 3.10.

.PARAMETER Clean
    Remove previous build artifacts (dist/, target/wheels/) first.

.PARAMETER Develop
    Install the freshly built extension into .venv via `maturin develop`
    instead of only producing a wheel (useful for local testing).

.EXAMPLE
    .\build_wheel.ps1

.EXAMPLE
    .\build_wheel.ps1 -Clean

.EXAMPLE
    .\build_wheel.ps1 -Develop
#>
[CmdletBinding()]
param(
    [string]$Python = "3.10",
    [switch]$Clean,
    [switch]$Develop
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

function Write-Step($msg) {
    Write-Host "==> $msg" -ForegroundColor Cyan
}

# 1. Ensure uv is available.
Write-Step "Checking uv"
uv --version | Out-Host

# 2. Optional clean.
if ($Clean) {
    Write-Step "Cleaning previous artifacts"
    foreach ($p in @("dist", "target\wheels")) {
        if (Test-Path $p) {
            Remove-Item -Recurse -Force $p
            Write-Host "    removed $p"
        }
    }
}

# 3. Create the virtual environment if missing.
$venv = Join-Path $ProjectRoot ".venv"
if (-not (Test-Path $venv)) {
    Write-Step "Creating virtual environment (.venv, Python $Python)"
    uv venv --python $Python $venv
}
else {
    Write-Step "Reusing existing .venv"
}

# 4. Install the build backend.
Write-Step "Installing build backend (maturin)"
uv pip install "maturin>=1.0,<2.0"

# 5. Build.
if ($Develop) {
    Write-Step "Building + installing into .venv (maturin develop --release)"
    uv run maturin develop --release
    Write-Host "`nInstalled wc_xpu into .venv (editable/native)." -ForegroundColor Green
}
else {
    Write-Step "Building release wheel (maturin build --release)"
    uv run maturin build --release --out dist
    Write-Host "`nWheel(s) written to dist\:" -ForegroundColor Green
    Get-ChildItem dist -Filter *.whl | ForEach-Object { Write-Host "    $($_.Name)" }
    Write-Host "`nInstall with:" -ForegroundColor Yellow
    Write-Host "    uv pip install (Get-ChildItem dist\*.whl | Select-Object -Last 1).FullName"
}
