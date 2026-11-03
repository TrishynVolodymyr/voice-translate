@echo off
chcp 65001 >nul
rem Step 2: downloads the models into models\ and checks their SHA-256.
rem   Download-Models.bat                core (~8 GB)
rem   Download-Models.bat -Llm           + Hy-MT2-7B translator and llama.cpp (~5 GB, used by Start-VoiceTranslate.bat)
rem   Download-Models.bat -Small         + smaller 0.6B voice
rem   Download-Models.bat -VoiceDesign   + model that creates a voice from a text description (Design-Voice.bat)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Download-Models.ps1" %*
pause
