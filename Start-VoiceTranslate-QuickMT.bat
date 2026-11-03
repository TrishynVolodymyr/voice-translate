@echo off
chcp 65001 >nul
rem Push-to-talk with the fast quickmt translator: recognition, translation and voice on the GPU (~8.5 GB VRAM).
rem Hold Right Ctrl, speak Ukrainian, release - hear English in your voice. Exit: q + Enter.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Start-VoiceTranslate.ps1" --mt quickmt %*
pause
