# Starts the push-to-talk Ukrainian -> English voice bridge. Arguments are passed to voice_translate.py
# (e.g. --tts 0.6B --mt quickmt --mode icl --key f9 --out 17 --ngl 28).
# For an LLM translator (default Hy-MT2-7B) this script also starts the folder's own llama-server on
# 127.0.0.1:8093 (VT_LLAMA_PORT) with -ngl from --ngl (default 999 = all layers on GPU) and stops it on exit.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root '.venv\Scripts\python.exe'
$server = Join-Path $root 'runtime\llama-server.exe'
# Settings from .env next to the launchers (see .env.example); variables already set in the environment win.
$envFile = Join-Path $root '.env'
if (Test-Path $envFile) {
    foreach ($line in Get-Content $envFile -Encoding UTF8) {
        if ($line -notmatch '^\s*#' -and $line -match '^\s*([A-Za-z_]\w*)\s*=\s*(.*?)\s*$' -and
            -not [Environment]::GetEnvironmentVariable($Matches[1])) {
            Set-Item "env:$($Matches[1])" $Matches[2].Trim('"')
        }
    }
}
$models = if (-not $env:VT_MODELS_DIR) { Join-Path $root 'models' }
          elseif ([IO.Path]::IsPathRooted($env:VT_MODELS_DIR)) { $env:VT_MODELS_DIR }
          else { Join-Path $root $env:VT_MODELS_DIR }
$port = if ($env:VT_LLAMA_PORT) { [int]$env:VT_LLAMA_PORT } else { 8093 }
$argline = $args -join ' '
# The last --mt wins (as in argparse): Start-StreamTest.bat passes "--mt quickmt" before the user's own arguments.
$mtAll = [regex]::Matches($argline, '--mt\s+(\S+)')
$mt = if ($mtAll.Count) { $mtAll[$mtAll.Count - 1].Groups[1].Value } else { 'hymt7' }
$ngl = if ($argline -match '--ngl\s+(\S+)') { $Matches[1] } else { '999' }
$ggufs = @{ 'hymt7' = 'hy-mt2\Hy-MT2-7B-Q4_K_M.gguf'; 'hymt1.8' = 'hy-mt2\Hy-MT2-1.8B-Q8_0.gguf';
            'gemma' = 'translategemma-4b-it\translategemma-4b-it.Q6_K.gguf' }
$useLlm = $ggufs.ContainsKey($mt)
if ($useLlm) { $gguf = Join-Path $models $ggufs[$mt] }

if (-not (Test-Path $py)) {
    Write-Host 'Середовище ще не встановлено: спершу запустіть Install.bat, потім Download-Models.bat (див. README).' -ForegroundColor Red
    exit 1
}
$need = @(
    (Join-Path $models 'silero-vad\silero_vad.onnx'),
    (Join-Path $models 'quickmt-uk-en\model.bin'),
    (Join-Path $models 'Qwen3-TTS-12Hz-1.7B-Base\model.safetensors')
)
if ($useLlm) { $need += $server, $gguf, (Join-Path $models 'parakeet-ultra-int8\encoder.int8.onnx') }  # LLM => ASR on the CPU: Parakeet Ultra
else { $need += Join-Path $models 'parakeet-tdt-0.6b-v3-fp32\encoder.weights' }  # quickmt => ASR on the GPU (--asr-device auto)
foreach ($f in $need) {
    if (-not (Test-Path $f)) {
        $hint = if ($useLlm) { 'Download-Models.bat -Llm' } else { 'Download-Models.bat' }
        Write-Host "Не знайдено: $f`nЗапустіть $hint (див. README)." -ForegroundColor Red
        exit 1
    }
}

# A window closed with the X button leaves its llama-server running without a parent: stop such leftovers of this
# folder (they hold ~2–5 GB of VRAM). Servers whose launcher window is still open are left alone.
Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" |
    Where-Object { $_.ExecutablePath -eq $server -and -not (Get-Process -Id $_.ParentProcessId -ErrorAction SilentlyContinue) } |
    ForEach-Object { Write-Host "Зупиняю залишений llama-server (PID $($_.ProcessId))"; Stop-Process -Id $_.ProcessId -Force }

# --cable (Start-VoiceTranslate-Cable.bat): the voice also goes into VB-CABLE, and while the translator runs the
# Windows default microphone is «CABLE Output», so games that cannot choose a microphone hear the English voice.
# The previous default is restored on exit; if the window was closed with X, on the next start (logs\default-mic.txt).
$cable = $args -contains '--cable'
$micState = Join-Path $root 'logs\default-mic.txt'
$prevMic = $null
if ($cable -or (Test-Path $micState)) { . (Join-Path $PSScriptRoot 'AudioDefault.ps1') }
if (Test-Path $micState) {
    $id = (Get-Content $micState -Raw).Trim()
    try { [VT.AudioDevices2]::SetDefault($id); Write-Host 'Повернуто мікрофон за замовчуванням після минулого запуску' } catch {}
    Remove-Item $micState
}

$llama = $null
try {
    if ($useLlm) {
        if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
            throw "Порт $port зайнятий — закрийте іншу копію перекладача або запустіть з --mt quickmt"
        }
        $llamaArgs = @('-m', $gguf, '--host', '127.0.0.1', '--port', $port, '-ngl', $ngl, '-c', '1024',
                       '-fa', 'on', '-t', '6', '--jinja', '--no-webui')
        Write-Host "Перекладач $mt ($($ggufs[$mt]), -ngl $ngl): http://127.0.0.1:$port"
        $llama = Start-Process -FilePath $server -PassThru -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $root 'logs\llama-server.log') `
            -RedirectStandardError (Join-Path $root 'logs\llama-server.err.log') -ArgumentList $llamaArgs
        $ready = $false
        foreach ($i in 1..60) {
            if ($llama.HasExited) { throw 'llama-server завершився — див. logs\llama-server.err.log' }
            try { if ((Invoke-RestMethod "http://127.0.0.1:$port/health" -TimeoutSec 2).status -eq 'ok') { $ready = $true; break } } catch {}
            Start-Sleep 1
        }
        if (-not $ready) { throw 'llama-server не відповів за 60 с — див. logs\llama-server.err.log' }
    }
    & nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
    $env:HF_HUB_OFFLINE = '1'
    $env:HF_HUB_CACHE = Join-Path $root 'cache\hf'
    $env:PYTHONUTF8 = '1'
    # --stream-test (from Start-StreamTest.bat) runs the text-only simultaneous translation test instead.
    $script = if ($args -contains '--stream-test') { 'stream_test.py' } else { 'voice_translate.py' }
    $pass = @($args | Where-Object { $_ -ne '--stream-test' })
    if ($cable) {
        $prevMic = Get-DefaultMic
        if ($prevMic.Name -like '*CABLE*') { throw 'Мікрофон за замовчуванням уже «CABLE Output» — поверніть свій у налаштуваннях звуку Windows' }
        Set-Content -Path $micState -Value $prevMic.Id -Encoding ascii
        Set-DefaultMic 'CABLE Output' | Out-Null
        Write-Host "Мікрофон за замовчуванням тепер «CABLE Output» (для ігор); перекладач слухає «$($prevMic.Name)»."
        Write-Host 'Виходьте через q + Enter — тоді Windows одразу поверне ваш мікрофон.'
        # MME device names are cut to 31 characters: the start of the name is enough to find the microphone
        if (-not ($args -contains '--mic')) { $pass += '--mic', $prevMic.Name.Substring(0, [Math]::Min(25, $prevMic.Name.Length)) }
    }
    & $py (Join-Path $root "scripts\$script") @pass
}
finally {
    if ($llama -and -not $llama.HasExited) { Stop-Process -Id $llama.Id -Force }
    if ($prevMic) {
        [VT.AudioDevices2]::SetDefault($prevMic.Id)
        Remove-Item $micState -ErrorAction SilentlyContinue
        Write-Host "Мікрофон за замовчуванням знову «$($prevMic.Name)»."
    }
}
