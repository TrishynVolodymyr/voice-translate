@echo off
chcp 65001 >nul
rem Same as Start-StreamTest.bat, but everything on the CPU: Parakeet Ultra int8 + quickmt (2 threads), no VRAM.
rem Fine on a free CPU; falls behind when games or other programs load the CPU (see README). Exit: q + Enter.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Start-VoiceTranslate.ps1" --stream-test --mt quickmt --asr-device cpu %*
pause
