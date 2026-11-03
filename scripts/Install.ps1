# Builds the voice-translate environment inside this folder — nothing is installed system-wide:
#   tools\uv\          uv (Python package manager), official release, SHA-256 checked
#   python\            Python 3.11 managed by uv
#   .venv\             PyTorch CUDA 12.8, Qwen3-TTS, sherpa-onnx (CUDA), CTranslate2, audio I/O (requirements.txt)
#   cache\uv\          package cache (can be deleted afterwards)
# Re-running it is safe: finished steps are skipped.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = Split-Path -Parent $PSScriptRoot
$dl = Join-Path $root 'downloads'
New-Item -ItemType Directory -Force $dl | Out-Null

function Get-Checked([string]$Url, [string]$Sha256) {
    $file = Join-Path $dl ([Uri]::UnescapeDataString(($Url -split '/')[-1]))
    if (-not (Test-Path $file) -or (Get-FileHash $file -Algorithm SHA256).Hash -ne $Sha256) {
        Write-Host "Завантажую $Url"
        Invoke-WebRequest -Uri $Url -OutFile $file -UseBasicParsing
        $got = (Get-FileHash $file -Algorithm SHA256).Hash
        if ($got -ne $Sha256) { Remove-Item $file; throw "Контрольна сума не збігається: $file ($got)" }
    }
    $file
}

# 1. uv
$uv = Join-Path $root 'tools\uv\uv.exe'
if (-not (Test-Path $uv)) {
    $zip = Get-Checked 'https://github.com/astral-sh/uv/releases/download/0.13.6/uv-x86_64-pc-windows-msvc.zip' `
                       'A2A8820D5EEA32F08EF467EF3B7F66883A9B076D5737A2FC2FBB5F458D1BB50F'
    Expand-Archive $zip -DestinationPath (Join-Path $root 'tools\uv') -Force
}
$env:UV_CACHE_DIR = Join-Path $root 'cache\uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $root 'python'
$env:UV_PYTHON_INSTALL_BIN = '0'
$env:UV_PYTHON_INSTALL_REGISTRY = '0'
$env:UV_NO_CONFIG = '1'

# 2. Python 3.11 and the virtual environment
& $uv python install 3.11
if ($LASTEXITCODE) { throw 'uv python install не вдався' }
$py = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) {
    & $uv venv (Join-Path $root '.venv') --python 3.11 --managed-python
    if ($LASTEXITCODE) { throw 'uv venv не вдався' }
}

# 3. Packages (PyTorch from its CUDA 12.8 index)
& $uv pip install --python $py -r (Join-Path $root 'requirements.txt') `
    --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
if ($LASTEXITCODE) { throw 'Встановлення пакетів не вдалося' }

# 4. sherpa-onnx with CUDA (the PyPI wheel runs on the CPU only); listed at https://k2-fsa.github.io/sherpa/onnx/cuda.html
$whl = Get-Checked 'https://huggingface.co/csukuangfj2/sherpa-onnx-wheels/resolve/main/cuda/2.0.1/sherpa_onnx-2.0.1+cuda12.cudnn9-cp311-cp311-win_amd64.whl' `
                   'C55A39F676835B57FE139863D69CE1A15B183829631578E087AA9BBDD9AE4642'
& $uv pip install --python $py $whl
if ($LASTEXITCODE) { throw 'Встановлення sherpa-onnx не вдалося' }

# 5. Check
& $py -c "import torch, transformers, sherpa_onnx, ctranslate2, faster_qwen3_tts; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'); print('transformers', transformers.__version__, '| sherpa-onnx', sherpa_onnx.__version__, '| ctranslate2', ctranslate2.__version__)"
if ($LASTEXITCODE) { throw 'Перевірка середовища не вдалася' }
Write-Host ''
Write-Host 'Середовище готове. Далі: Download-Models.bat'
