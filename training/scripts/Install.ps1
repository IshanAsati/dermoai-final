$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $ProjectRoot
$env:UV_CACHE_DIR = Join-Path $ProjectRoot 'runtime\cache\uv'
uv python install 3.12
if ($LASTEXITCODE -ne 0) { throw 'Python installation failed.' }
uv venv --python 3.12 .venv
if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed.' }
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
uv pip install --python $Python torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
uv pip install --python $Python -e . --constraint requirements-tested.txt
if ($LASTEXITCODE -ne 0) { throw 'Integration dependency installation failed.' }
uv pip install --python $Python --no-deps 'label-studio-ml @ git+https://github.com/HumanSignal/label-studio-ml-backend.git@7b0d01f3593e327ad9cd2a92dbd08aa89f6403f4'
if ($LASTEXITCODE -ne 0) { throw 'Official ML backend installation failed.' }
. "$PSScriptRoot\Environment.ps1"
& $Python -m dermoai.cli health
