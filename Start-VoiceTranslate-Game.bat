@echo off
chcp 65001 >nul
rem Push-to-talk next to a demanding game: recognition and translation on the CPU, the 1.7B voice on the GPU (~5 GB VRAM);
rem the translated voice goes into VB-CABLE and your headphones. If the voice starts to stutter in heavy scenes, use the
rem lighter voice: Start-VoiceTranslate-Game.bat --tts 0.6B (needs Download-Models.bat -Small).
rem Tested in Path of Exile 2: first sound 0.6-1.0 s (1.7B) / 0.55-1.3 s (0.6B), game 80-90 FPS (130-145 without).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Start-VoiceTranslate.ps1" --mt-default quickmt --asr-device cpu --cable --ngl-default 0 %*
pause
