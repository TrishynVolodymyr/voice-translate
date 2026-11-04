@echo off
chcp 65001 >nul
rem Push-to-talk with the most accurate translator, Hy-MT2-7B (needs Download-Models.bat -Llm); recognition on the CPU.
rem Hold Right Ctrl, speak, release - hear the translation in your voice (languages: .env). Exit: q + Enter.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Start-VoiceTranslate.ps1" %*
pause
