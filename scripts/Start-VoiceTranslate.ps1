# Starts the push-to-talk voice bridge (Ukrainian -> English by default; languages and translator in .env). Arguments
# are passed to voice_translate.py (e.g. --tts 0.6B --mt quickmt --mode icl --key f9 --out 17 --ngl 28).
# For an LLM translator (default Hy-MT2-7B, or any GGUF in VT_MT) this script also starts the folder's own llama-server on
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
# The last value wins (as in argparse): a launcher's own arguments come before the user's (%*).
function Get-LastArg([string]$name) { $all = [regex]::Matches($argline, "--$name\s+(\S+)"); if ($all.Count) { $all[$all.Count - 1].Groups[1].Value } }
$ngl = Get-LastArg 'ngl'; if (-not $ngl) { $ngl = '999' }
# Translator, in this order: --mt typed by the user > VT_MT in .env > the launcher's own --mt-default > Hy-MT2-7B.
# A value is a built-in name or the path to any GGUF (e.g. your own fine-tune), served by llama-server with its chat
# template and prompted like Hy-MT2.
$ggufs = @{ 'hymt7' = 'hy-mt2\Hy-MT2-7B-Q4_K_M.gguf'; 'hymt1.8' = 'hy-mt2\Hy-MT2-1.8B-Q8_0.gguf';
            'gemma' = 'translategemma-4b-it\translategemma-4b-it.Q6_K.gguf' }
$mt = Get-LastArg 'mt'
if (-not $mt) { $mt = $env:VT_MT }
if (-not $mt) { $mt = Get-LastArg 'mt-default' }
if (-not $mt) { $mt = 'hymt7' }
$gguf = $null
if ($mt -like '*.gguf') {
    $gguf = if ([IO.Path]::IsPathRooted($mt)) { $mt } else { Join-Path $root $mt }
    $mtName = 'gguf'
} else {
    if ($ggufs.ContainsKey($mt)) { $gguf = Join-Path $models $ggufs[$mt] }
    $mtName = $mt
}
$useLlm = [bool]$gguf
$src = if ($env:VT_SOURCE_LANG) { $env:VT_SOURCE_LANG.ToLower() } else { 'uk' }
$tgt = if ($env:VT_TARGET_LANG) { $env:VT_TARGET_LANG.ToLower() } else { 'en' }
if ($mt -eq 'quickmt' -and -not (Test-Path (Join-Path $models "quickmt-$src-$tgt\model.bin"))) {
    Write-Host ("Для пари $src→$tgt немає моделі quickmt (вони є лише з англійської та на англійську, і їх завантажує " +
                "Download-Models.bat). Вкажіть інший перекладач у .env: VT_MT=hymt7, VT_MT=hymt1.8 або шлях до свого " +
                "GGUF — див. README, «Мови».") -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $py)) {
    Write-Host 'Середовище ще не встановлено: спершу запустіть Install.bat, потім Download-Models.bat (див. README).' -ForegroundColor Red
    exit 1
}
# Only the models the chosen mode needs: voice size (--tts) and where recognition runs (--asr-device; auto = on the
# GPU with quickmt, on the CPU with an LLM translator).
$tts = Get-LastArg 'tts'; if (-not $tts) { $tts = '1.7B' }
$asrDev = Get-LastArg 'asr-device'; if (-not $asrDev -or $asrDev -eq 'auto') { $asrDev = if ($useLlm) { 'cpu' } else { 'cuda' } }
$need = @(
    (Join-Path $models 'silero-vad\silero_vad.onnx'),
    (Join-Path $models "Qwen3-TTS-12Hz-$tts-Base\model.safetensors")
)
# quickmt: one model per language pair (VT_SOURCE_LANG / VT_TARGET_LANG); its absence is explained by the Python side
if ($useLlm) { $need += $server, $gguf }
if ($asrDev -eq 'cpu') { $need += Join-Path $models 'parakeet-ultra-int8\encoder.int8.onnx' }
else { $need += Join-Path $models 'parakeet-tdt-0.6b-v3-fp32\encoder.weights' }
foreach ($f in $need) {
    if (-not (Test-Path $f)) {
        $hint = if ($f -eq $gguf -and $mtName -eq 'gguf') { 'Перевірте шлях VT_MT у .env' }
                elseif ($f -eq $gguf -and $mt -eq 'hymt7') { 'Запустіть Download-Models.bat -Llm' }
                elseif ($tts -eq '0.6B' -and $f -like '*0.6B*') { 'Запустіть Download-Models.bat -Small' }
                else { 'Запустіть Download-Models.bat' }
        Write-Host "Не знайдено: $f`n$hint (див. README)." -ForegroundColor Red
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
        Write-Host "Перекладач $(Split-Path -Leaf $gguf) (-ngl $ngl): http://127.0.0.1:$port"
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
    # The translator resolved above replaces every --mt / --mt-default (Python knows a custom GGUF as "gguf")
    $pass = @()
    for ($i = 0; $i -lt $args.Count; $i++) {
        if ($args[$i] -in '--mt', '--mt-default') { $i++; continue }
        if ($args[$i] -ne '--stream-test') { $pass += $args[$i] }
    }
    $pass += '--mt', $mtName
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
