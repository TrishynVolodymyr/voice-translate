# Downloads the models into models\ (or VT_MODELS_DIR) and checks SHA-256 of every file.
# Hugging Face caches go to cache\ (deleted at the end); no account or token is needed — all models are public.
#   Download-Models.ps1            core: recognition, translation (quickmt), voice 1.7B, voice activity detection (~8 GB)
#   Download-Models.ps1 -Llm       + Hy-MT2-7B translator and llama.cpp b12040 runtime (~5 GB; default of Start-VoiceTranslate.bat)
#   Download-Models.ps1 -Small     + Qwen3-TTS 0.6B voice (~2.4 GB; less VRAM, less natural)
#   Download-Models.ps1 -VoiceDesign  + Qwen3-TTS VoiceDesign (~4.2 GB; only to create a voice from a text description)
param([switch]$Llm, [switch]$Small, [switch]$VoiceDesign)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root '.venv\Scripts\python.exe'
$hf = Join-Path $root '.venv\Scripts\hf.exe'
if (-not (Test-Path $hf)) { throw 'Спершу запустіть Install.bat' }
# Settings from .env (models folder, language pair); variables already set in the environment win.
$envFile = Join-Path $root '.env'
if (Test-Path $envFile) {
    foreach ($line in Get-Content $envFile -Encoding UTF8) {
        if ($line -notmatch '^\s*#' -and $line -match '^\s*([A-Za-z_]\w*)\s*=\s*(.*?)\s*$' -and
            -not [Environment]::GetEnvironmentVariable($Matches[1])) {
            Set-Item "env:$($Matches[1])" $Matches[2].Trim('"')
        }
    }
}
$src = if ($env:VT_SOURCE_LANG) { $env:VT_SOURCE_LANG.ToLower() } else { 'uk' }
$tgt = if ($env:VT_TARGET_LANG) { $env:VT_TARGET_LANG.ToLower() } else { 'en' }
$m =if ($env:VT_MODELS_DIR) { $env:VT_MODELS_DIR } else { Join-Path $root 'models' }
if (-not [IO.Path]::IsPathRooted($m)) { $m = Join-Path $root $m }
$env:HF_HUB_CACHE = Join-Path $root 'cache\hf'
$env:HF_XET_CACHE = Join-Path $root 'cache\xet'
$env:HF_HUB_DISABLE_TELEMETRY = '1'

function Get-HF([string]$Repo, [string]$Dir, [string[]]$Files) {
    Write-Host "== $Repo"
    & $hf download $Repo @Files --local-dir (Join-Path $m $Dir)
    if ($LASTEXITCODE) { throw "Не вдалося завантажити $Repo" }
}

function Get-Checked([string]$Url, [string]$Sha256, [string]$Dir) {
    New-Item -ItemType Directory -Force $Dir | Out-Null
    $file = Join-Path $Dir (($Url -split '/')[-1])
    if (-not (Test-Path $file) -or (Get-FileHash $file -Algorithm SHA256).Hash -ne $Sha256) {
        Write-Host "== $Url"
        Invoke-WebRequest -Uri $Url -OutFile $file -UseBasicParsing
        if ((Get-FileHash $file -Algorithm SHA256).Hash -ne $Sha256) { Remove-Item $file; throw "Контрольна сума не збігається: $file" }
    }
    $file
}

# Core
Get-Checked 'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx' `
            '9E2449E1087496D8D4CABA907F23E0BD3F78D91FA552479BB9C23AC09CBB1FD6' (Join-Path $m 'silero-vad') | Out-Null
Get-HF 'chrmongeau/sherpa-onnx-nemo-parakeet-ultra-int8' 'parakeet-ultra-int8' @('encoder.int8.onnx', 'decoder.int8.onnx', 'joiner.int8.onnx', 'tokens.txt')
Get-HF 'csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3' 'parakeet-tdt-0.6b-v3-fp32' @('encoder.onnx', 'encoder.weights', 'decoder.onnx', 'joiner.onnx', 'tokens.txt')
# quickmt has a separate model for each pair, only from and into English (quickmt-uk-en, quickmt-en-de, …)
$qm = "quickmt-$src-$tgt"
$hasQm = $true
try { Invoke-RestMethod "https://huggingface.co/api/models/quickmt/$qm" | Out-Null } catch { $hasQm = $false }
if ($hasQm) {
    Get-HF "quickmt/$qm" $qm @('config.json', 'model.bin', 'source_vocabulary.json', 'target_vocabulary.json', 'src.spm.model', 'tgt.spm.model', 'README.md')
} else {
    Write-Host "Для пари $src→$tgt немає моделі quickmt — перекладатиме Hy-MT2-7B (Download-Models.bat -Llm, Start-VoiceTranslate.bat)." -ForegroundColor Yellow
}
Get-HF 'Qwen/Qwen3-TTS-12Hz-1.7B-Base' 'Qwen3-TTS-12Hz-1.7B-Base' @()

if ($Small) { Get-HF 'Qwen/Qwen3-TTS-12Hz-0.6B-Base' 'Qwen3-TTS-12Hz-0.6B-Base' @() }
if ($VoiceDesign) { Get-HF 'Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign' 'Qwen3-TTS-12Hz-1.7B-VoiceDesign' @() }
if ($Llm) {
    Get-HF 'tencent/Hy-MT2-7B-GGUF' 'hy-mt2' @('Hy-MT2-7B-Q4_K_M.gguf')
    $rt = Join-Path $root 'runtime'
    if (-not (Test-Path (Join-Path $rt 'llama-server.exe'))) {
        $dl = Join-Path $root 'downloads'
        $a = Get-Checked 'https://github.com/ggml-org/llama.cpp/releases/download/b12040/llama-b12040-bin-win-cuda-12.4-x64.zip' `
                         '71472D2206A8E1618E3902F598DEC77FE7AEE74C5B75159E3D46CBB749361A62' $dl
        $b = Get-Checked 'https://github.com/ggml-org/llama.cpp/releases/download/b12040/cudart-llama-bin-win-cuda-12.4-x64.zip' `
                         '8C79A9B226DE4B3CACFD1F83D24F962D0773BE79F1E7B75C6AF4DED7E32AE1D6' $dl
        Expand-Archive $a -DestinationPath $rt -Force
        Expand-Archive $b -DestinationPath $rt -Force
    }
}

Remove-Item -Recurse -Force (Join-Path $root 'cache\hf'), (Join-Path $root 'cache\xet') -ErrorAction SilentlyContinue
Get-ChildItem $m -Recurse -Directory -Filter '.cache' | Remove-Item -Recurse -Force
& $py (Join-Path $PSScriptRoot 'verify_sha.py')
if ($LASTEXITCODE) { throw 'Перевірка контрольних сум не пройшла' }
Write-Host ''
Write-Host 'Моделі готові. Далі: запишіть свій голос (Record-Samples.bat) або створіть голос за описом (Design-Voice.bat).'
