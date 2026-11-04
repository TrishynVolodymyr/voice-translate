@echo off
chcp 65001 >nul
rem Push-to-talk: hold Right Ctrl, speak, release - the translation sounds in your voice in your headphones and, if
rem VB-CABLE is installed, in the «CABLE Output» virtual microphone for Discord, calls and games (it is the Windows
rem default microphone while this runs; yours comes back on exit). Recognition and translation on the CPU, the voice
rem on the GPU (~5 GB VRAM). Languages, translator (VT_MT) and the rest: .env. Exit: q + Enter or close the window.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Start-VoiceTranslate.ps1" --mt-default quickmt %*
pause
