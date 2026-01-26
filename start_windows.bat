@echo off
REM LocalMind Windows startup script with enhanced crash prevention

echo Starting LocalMind for Windows...

REM Check for virtual environment
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo Virtual environment activated
) else (
    echo Warning: Using system Python
)

REM Install Windows-compatible dependencies
echo Installing Windows-compatible dependencies...
pip uninstall -y llama-cpp-python 2>nul
pip install llama-cpp-python --force-reinstall --no-cache-dir
pip install -r requirements.txt

REM Set Windows-specific environment variables
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

REM Fix any corrupted files
if exist fix_crashes.py (
    echo Checking for corrupted files...
    python fix_crashes.py
)

REM Launch LocalMind with error handling
echo Launching LocalMind...
python main.py

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo Application encountered an error. Check the logs folder for details.
    echo Press any key to exit...
    pause >nul
)
