@echo off
chcp 65001 >nul
rem Simultaneous translation with voice: speak without pauses, the translation in your cloned voice follows ~2-4 s behind.
rem Parakeet + quickmt + Qwen3-TTS 1.7B, all on the GPU (~8.5 GB VRAM); also into VB-CABLE if installed (VT_CABLE=0: off).
rem Use headphones. Exit: q + Enter.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Start-VoiceTranslate.ps1" --stream-test --mt-default quickmt --voice %*
pause
