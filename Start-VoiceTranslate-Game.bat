@echo off
chcp 65001 >nul
rem Push-to-talk for games: light mode that fits next to a demanding game - voice 0.6B on the GPU (~3 GB VRAM),
rem recognition and translation on the CPU; the English voice goes into VB-CABLE and your headphones.
rem Needs Download-Models.bat -Small. Tested in Path of Exile 2: first sound 0.55-1.3 s, game 80-90 FPS (130-145 without).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Start-VoiceTranslate.ps1" --mt quickmt --tts 0.6B --asr-device cpu --cable %*
pause
