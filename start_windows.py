#!/usr/bin/env python3
"""
Windows-specific startup script for LocalMind
Handles Windows compatibility and dependency checks
"""

import sys
import os
import platform
import subprocess

def check_python_version():
    """Ensure Python 3.8+ is being used"""
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    return True

def install_dependencies():
    """Install required dependencies if missing"""
    required_packages = [
        "PySide6>=6.5.0",
        "llama-cpp-python>=0.2.0",
        "PyPDF2>=3.0.0", 
        "python-docx>=0.8.11"
    ]
    
    print("Checking dependencies...")
    
    for package in required_packages:
        try:
            # Try importing the package
            pkg_name = package.split(">=")[0].replace("-", "_")
            if pkg_name == "python_docx":
                pkg_name = "docx"
            elif pkg_name == "llama_cpp_python":
                pkg_name = "llama_cpp"
            
            __import__(pkg_name)
            print(f"✓ {package}")
            
        except ImportError:
            print(f"✗ {package} - Installing...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    package, "--force-reinstall", "--no-cache-dir"
                ])
                print(f"✓ {package} installed")
            except subprocess.CalledProcessError:
                print(f"✗ Failed to install {package}")
                return False
    
    return True

def setup_windows_environment():
    """Configure Windows-specific environment"""
    if platform.system() != "Windows":
        return
    
    # Set encoding
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    # DPI handling
    os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'
    
    # Check Windows version for specific settings
    try:
        version = platform.version()
        build = int(version.split('.')[-1]) if '.' in version else 0
        if build >= 22000:  # Windows 11
            os.environ['QT_ENABLE_HIGHDPI_SCALING'] = '1'
            os.environ['QT_SCALE_FACTOR_ROUNDING_POLICY'] = 'RoundPreferFloor'
            print("Windows 11 detected - Enhanced DPI support enabled")
        else:  # Windows 10 and older
            os.environ['QT_ENABLE_HIGHDPI_SCALING'] = '0'
            print("Windows 10/older detected - Conservative DPI settings")
    except:
        # Fallback to conservative settings
        os.environ['QT_ENABLE_HIGHDPI_SCALING'] = '0'

def main():
    """Main startup function"""
    print("LocalMind Windows Startup")
    print("=" * 30)
    
    # Check Python version
    if not check_python_version():
        input("Press Enter to exit...")
        sys.exit(1)
    
    # Setup Windows environment
    setup_windows_environment()
    
    # Install dependencies
    if not install_dependencies():
        print("\nFailed to install required dependencies")
        input("Press Enter to exit...")
        sys.exit(1)
    
    print("\nStarting LocalMind...")
    
    # Import and run main application
    try:
        from main import main as app_main
        app_main()
    except Exception as e:
        print(f"\nError starting LocalMind: {e}")
        input("Press Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()
