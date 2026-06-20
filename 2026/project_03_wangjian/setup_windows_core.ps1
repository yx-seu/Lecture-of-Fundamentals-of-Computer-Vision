$ErrorActionPreference = "Stop"

Write-Host "Project 03 core environment setup" -ForegroundColor Cyan
Write-Host "This installs the packages needed for main.py, YOLOv11, SAM 2, and the notebook." -ForegroundColor Cyan

$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""

$venvRoot = "C:\cv03_env"
if (-not (Test-Path $venvRoot)) {
    py -3.13 -m venv $venvRoot
}

$venvPython = Join-Path $venvRoot "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Cannot find Python inside $venvRoot."
}

& $venvPython -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --trusted-host download.pytorch.org
& $venvPython -m pip install --prefer-binary numpy pandas pillow opencv-python matplotlib requests tqdm pyyaml -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
& $venvPython -m pip install --prefer-binary ultralytics -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
& $venvPython -m pip install https://github.com/facebookresearch/sam2/archive/refs/heads/main.zip

Write-Host ""
Write-Host "Core setup finished. Now run:" -ForegroundColor Green
Write-Host "  C:\cv03_env\Scripts\Activate.ps1"
Write-Host "  python src\check_env.py"
Write-Host "  python src\main.py --input data\test_examples --output results"

