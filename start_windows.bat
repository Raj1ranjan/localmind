@echo off
REM Windows startup script for LocalMind
REM Fixes common Windows issues before launching

echo Starting LocalMind...

REM Fix corrupted chat files first
echo Checking for corrupted chat files...
python fix_crashes.py

REM Set environment variables for better Windows compatibility
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

REM Launch LocalMind
echo Launching LocalMind...
python main.py

pause
