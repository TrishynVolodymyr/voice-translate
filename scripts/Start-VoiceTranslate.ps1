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
# Layers of an LLM translator on the GPU: --ngl typed by the user > VT_MT_NGL in .env > 999 (all on the GPU).
$ngl = Get-LastArg 'ngl'
if (-not $ngl) { $ngl = $env:VT_MT_NGL }
if (-not $ngl) { $ngl = '999' }
# Translator, in this order: --mt typed by the user > VT_MT in .env > the launcher's own --mt-default > Hy-MT2-7B.
# A value is a built-in name or the path to any GGUF (e.g. your own fine-tune), served by llama-server with its chat
# template and prompted like Hy-MT2.
$ggufs = @{ 'hymt7' = 'hy-mt2\Hy-MT2-7B-Q4_K_M.gguf'; 'hymt1.8' = 'hy-mt2\Hy-MT2-1.8B-Q8_0.gguf' }
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
# Only the models the chosen mode needs: voice size (--tts) and where recognition runs.
# Recognition (and quickmt, which follows it): --asr-device typed by the user > VT_ASR_DEVICE in .env > auto.
# auto: push-to-talk on the CPU — Parakeet Ultra there is more accurate, the GPU saved only ~0.15 s per phrase
# (20 phrases: 0.61 vs 0.43 s to the first sound) and took ~3.5 GB of VRAM; the simultaneous mode re-reads the phrase
# every 0.5 s, which a busy CPU cannot keep up with, so there it stays on the GPU with quickmt.
$tts = Get-LastArg 'tts'; if (-not $tts) { $tts = '1.7B' }
$stream = $args -contains '--stream-test'
$asrDev = Get-LastArg 'asr-device'
if (-not $asrDev -or $asrDev -eq 'auto') { $asrDev = $env:VT_ASR_DEVICE }
if (-not $asrDev -or $asrDev -eq 'auto') { $asrDev = if ($stream -and -not $useLlm) { 'cuda' } else { 'cpu' } }
$need = @(
    (Join-Path $models 'silero-vad\silero_vad.onnx'),
    (Join-Path $models "Qwen3-TTS-12Hz-$tts-Base\model.safetensors")
)
# quickmt: one model per language pair (VT_SOURCE_LANG / VT_TARGET_LANG); its absence is explained by the Python side
if ($useLlm) { $need += $server, $gguf }
# Recognizer (--asr): auto = Parakeet Ultra on the CPU / Parakeet v3 fp32 on the GPU; parakeet = v3 (int8 on the CPU);
# hybrid = v3 plus the Ukrainian-only NeMo model for phrases v3 writes in Russian or Latin letters. v3 int8 and NeMo
# are not in the core download: Download-Models.bat -Hybrid.
$asrKind = Get-LastArg 'asr'
$v3 = if ($asrDev -eq 'cpu') { 'parakeet-tdt-0.6b-v3-int8\encoder.int8.onnx' } else { 'parakeet-tdt-0.6b-v3-fp32\encoder.weights' }
switch ($asrKind) {
    'ultra' { $need += Join-Path $models 'parakeet-ultra-int8\encoder.int8.onnx' }
    'parakeet' { $need += Join-Path $models $v3 }
    'hybrid' { $need += (Join-Path $models $v3), (Join-Path $models 'nemo-fastconformer-ctc-uk\model.onnx') }
    default {
        if ($asrDev -eq 'cpu') { $need += Join-Path $models 'parakeet-ultra-int8\encoder.int8.onnx' }
        else { $need += Join-Path $models 'parakeet-tdt-0.6b-v3-fp32\encoder.weights' }
    }
}
foreach ($f in $need) {
    if (-not (Test-Path $f)) {
        $hint = if ($f -eq $gguf -and $mtName -eq 'gguf') { 'Перевірте шлях VT_MT у .env' }
                elseif ($f -eq $gguf -and $mt -eq 'hymt7') { 'Запустіть Download-Models.bat -Llm' }
                elseif ($tts -eq '0.6B' -and $f -like '*0.6B*') { 'Запустіть Download-Models.bat -Small' }
                elseif ($f -like '*nemo-fastconformer*' -or $f -like '*v3-int8*') { 'Запустіть Download-Models.bat -Hybrid' }
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

# Virtual microphone: whenever a virtual cable is installed and the mode speaks, the voice also goes into it, and while
# the translator runs the Windows default microphone is the cable's recording side — so Discord, calls and games that
# cannot choose a microphone hear the translation. VB-CABLE by default; any other cable (VoiceMeeter, Virtual Audio
# Cable…) through VT_CABLE_PLAY (the playback device the voice goes into) and VT_CABLE_MIC (its recording side) in
# .env. VT_CABLE=0 turns it off (e.g. to talk to someone in the room through the speakers); --cable forces it on. The previous default is restored on exit (q + Enter); if the window is closed with X
# or Ctrl+C kills this script, a hidden watcher restores it once this process is gone; failing that, the next start
# does (logs\default-mic.txt).
$micState = Join-Path $root 'logs\default-mic.txt'
$prevMic = $null
$speaks = -not $stream -or ($args -contains '--voice')
$cableOff = "$env:VT_CABLE" -in '0', 'off', 'no', 'false'
$cablePlay = if ($env:VT_CABLE_PLAY) { $env:VT_CABLE_PLAY } else { 'CABLE Input' }
$cableMic = if ($env:VT_CABLE_MIC) { $env:VT_CABLE_MIC } else { 'CABLE Output' }
$cable = $false
if ($speaks -and -not $cableOff -or ($args -contains '--cable') -or (Test-Path $micState)) {
    . (Join-Path $PSScriptRoot 'AudioDefault.ps1')
}
if ($args -contains '--cable') { $cable = $true }
elseif ($speaks -and -not $cableOff) {
    $cable = [bool](Get-Mics | Where-Object { $_.Name -like "*$cableMic*" }) -and
             [bool](Get-Speakers | Where-Object { $_.Name -like "*$cablePlay*" })
    if (-not $cable -and -not $env:VT_CABLE_MIC -and -not $env:VT_CABLE_PLAY) {
        Write-Host ('Віртуальний мікрофон VB-CABLE не встановлено — переклад звучить лише у ваших навушниках. Щоб його ' +
                    'чули Discord, ігри й дзвінки, запустіть Setup-VirtualCable.bat (або вкажіть інший кабель у .env).') -ForegroundColor Yellow
    } elseif (-not $cable) {
        Write-Host ("Не знайдено віртуальний кабель «$cablePlay» → «$cableMic» (VT_CABLE_PLAY / VT_CABLE_MIC у .env) — " +
                    'переклад звучить лише у ваших навушниках.') -ForegroundColor Yellow
    }
}
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
        if ($ngl -eq '0') {
            Write-Host ('Перекладач працює на процесорі (повільніше: Hy-MT2-1.8B ~0,7–0,9 с на фразу замість ~0,1 с), щоб ' +
                        'відеопам''ять лишилася грі. Якщо вона у вас є (напр. 16 ГБ) — VT_MT_NGL=999 у .env.') -ForegroundColor Yellow
        }
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
    # The translator and -ngl resolved above replace every --mt / --ngl given (Python knows a custom GGUF as "gguf")
    $pass = @()
    for ($i = 0; $i -lt $args.Count; $i++) {
        if ($args[$i] -in '--mt', '--mt-default', '--ngl', '--asr-device') { $i++; continue }
        if ($args[$i] -notin '--stream-test', '--cable') { $pass += $args[$i] }
    }
    if ($cable) { $pass += '--cable' }
    # resolved values, so Python runs and shows exactly what was checked here
    $pass += '--mt', $mtName, '--ngl', $ngl, '--asr-device', $asrDev
    if ($cable) {
        $prevMic = Get-DefaultMic
        if ($prevMic.Name -like "*$cableMic*") { throw "Мікрофон за замовчуванням уже «$($prevMic.Name)» — поверніть свій у налаштуваннях звуку Windows" }
        Set-Content -Path $micState -Value $prevMic.Id -Encoding ascii
        # The watcher runs in its own hidden console, so closing this window does not kill it. It waits for this
        # PowerShell process to end and restores the microphone if the state file is still there (q + Enter removes
        # it in `finally` first, so then it does nothing).
        $watch = "`$p = Get-Process -Id $PID -ErrorAction SilentlyContinue; if (`$p) { `$p.WaitForExit() }; " +
                 "`$s = '$($micState -replace "'", "''")'; if (Test-Path `$s) { . '$((Join-Path $PSScriptRoot 'AudioDefault.ps1') -replace "'", "''")'; " +
                 "[VT.AudioDevices2]::SetDefault((Get-Content `$s -Raw).Trim()); Remove-Item `$s }"
        Start-Process powershell -WindowStyle Hidden -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass',
            '-EncodedCommand', [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($watch)))
        Set-DefaultMic $cableMic | Out-Null
        Write-Host "Мікрофон за замовчуванням тепер «$cableMic» (для Discord та ігор); перекладач слухає «$($prevMic.Name)»."
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
