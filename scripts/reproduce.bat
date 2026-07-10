@echo off
REM GPT4RUL one-click reproduction (double-click entry)
cd /d "%~dp0.."
powershell -ExecutionPolicy Bypass -File "%~dp0reproduce.ps1" %*
pause
