@echo off
cd /d "%~dp0"
call .venv\Scripts\python.exe wsp\live_subtitles.py %*
pause
