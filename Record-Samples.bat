@echo off
chcp 65001 >nul
rem Records your voice sample (and optional Ukrainian test phrases) into voices\ (VT_VOICES_DIR). Enter starts/stops.
rem Arguments: --list (audio devices), --device N, --redo, --with-en
"%~dp0.venv\Scripts\python.exe" -X utf8 "%~dp0scripts\record_samples.py" %*
pause
