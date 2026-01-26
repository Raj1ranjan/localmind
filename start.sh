#!/bin/bash
# LocalMind startup script

cd "$(dirname "$0")"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Activated virtual environment"
else
    echo "Error: .venv directory not found"
    exit 1
fi

# Check dependencies
python -c "import PySide6, llama_cpp" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Run application
echo "Starting LocalMind..."
python main.py
