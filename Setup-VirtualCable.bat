@echo off
chcp 65001 >nul
rem Helps to install VB-CABLE (virtual microphone for Discord and games): remembers your audio devices, opens the
rem official page, and after you install it switches your microphone and speakers back from the cable.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Setup-VirtualCable.ps1"
pause
