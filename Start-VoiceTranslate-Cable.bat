@echo off
chcp 65001 >nul
rem Push-to-talk for Discord and games: the translated voice goes into VB-CABLE (microphone «CABLE Output») and your
rem headphones. While it runs, «CABLE Output» is the Windows default microphone; your own is restored on exit (q + Enter).
rem quickmt + recognition + voice on the GPU (~8.5 GB VRAM). Hold Right Ctrl and speak.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Start-VoiceTranslate.ps1" --mt-default quickmt --cable %*
pause
