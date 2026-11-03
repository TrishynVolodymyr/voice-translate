@echo off
chcp 65001 >nul
rem Creates a voice from a text description instead of cloning your own (needs Download-Models.bat -VoiceDesign).
rem   Design-Voice.bat "A calm male voice in his thirties, low and slightly husky" --name calm
rem then: Start-VoiceTranslate.bat --ref calm   (or set VT_REF=calm in .env)
"%~dp0.venv\Scripts\python.exe" -X utf8 "%~dp0scripts\design_voice.py" %*
pause
