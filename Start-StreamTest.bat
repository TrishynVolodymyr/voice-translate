@echo off
chcp 65001 >nul
rem Text-only simultaneous translation test: speak without pauses, the translated text grows as you talk.
rem Parakeet + quickmt on the GPU (VT_ASR_DEVICE=cpu in .env: on the CPU); add --mt hymt7 for Hy-MT2-7B. Exit: q + Enter.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Start-VoiceTranslate.ps1" --stream-test --mt-default quickmt %*
pause
