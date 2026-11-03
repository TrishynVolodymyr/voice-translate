@echo off
chcp 65001 >nul
rem Step 1: builds Python, PyTorch (CUDA) and all packages inside this folder (~6 GB). Nothing is installed system-wide.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Install.ps1"
pause
