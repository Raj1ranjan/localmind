#!/usr/bin/env python3
"""
LocalMind Setup Script
Automated setup for LocalMind AI Chat Application
"""

import os
import sys
import subprocess
import platform

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"⏳ {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def main():
    print("🧠 LocalMind Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Create virtual environment
    if not os.path.exists('.venv'):
        if not run_command(f"{sys.executable} -m venv .venv", "Creating virtual environment"):
            sys.exit(1)
    
    # Determine activation command
    if platform.system() == "Windows":
        activate_cmd = ".venv\\Scripts\\activate"
        pip_cmd = ".venv\\Scripts\\pip"
        python_cmd = ".venv\\Scripts\\python"
    else:
        activate_cmd = "source .venv/bin/activate"
        pip_cmd = ".venv/bin/pip"
        python_cmd = ".venv/bin/python"
    
    # Install requirements
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing dependencies"):
        sys.exit(1)
    
    print("\n🎉 Setup Complete!")
    print("\nTo start LocalMind:")
    if platform.system() == "Windows":
        print("  python start_windows.py")
    else:
        print("  ./start.sh")
    print("  or")
    print(f"  {activate_cmd} && python main.py")

if __name__ == "__main__":
    main()
