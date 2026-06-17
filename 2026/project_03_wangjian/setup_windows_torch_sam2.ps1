$ErrorActionPreference = "Stop"

Write-Host "Installing PyTorch and SAM 2 optional dependencies" -ForegroundColor Cyan

$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""

$venvPython = "C:\cv03_env\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Cannot find Python inside C:\cv03_env. Run setup_windows_core.ps1 first."
}

& $venvPython -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --trusted-host download.pytorch.org
& $venvPython -m pip install labelme -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com

Write-Host ""
Write-Host "Optional setup finished. Download sam2.1_hiera_small.pt and place it in weights\." -ForegroundColor Green
