import os, sys

if hasattr(sys, "_MEIPASS"):
    os.environ["LD_LIBRARY_PATH"] = (
        sys._MEIPASS + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    )



#!/usr/bin/env python3
import sys
import logging
import os
from pathlib import Path

# Add site-packages to path for Flatpakpip
sys.path.insert(0, '/app/lib/python3.11/site-packages')

# Configure logging with file output for debugging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "localmind.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Check Python version
        if sys.version_info < (3, 8):
            raise RuntimeError("Python 3.8 or higher is required")
        
        # Import PySide6 with error handling
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox
        except ImportError as e:
            error_msg = f"PySide6 not found: {str(e)}"
            logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            print("Please install PySide6: pip install PySide6")
            sys.exit(1)
        
        # Create QApplication
        try:
            app = QApplication(sys.argv)
            app.setStyle('Fusion')
        except Exception as e:
            logger.error(f"Error creating QApplication: {e}")
            print(f"ERROR: Could not create application: {e}")
            sys.exit(1)
        
        # Import UI after QApplication is created to avoid potential issues
        try:
            from ui.main_window import MainWindow
        except ImportError as e:
            error_msg = f"Missing UI dependencies: {str(e)}"
            logger.error(error_msg)
            QMessageBox.critical(None, "Dependency Error", 
                               f"{error_msg}\n\nPlease install required packages:\n"
                               f"pip install PySide6 llama-cpp-python")
            sys.exit(1)
        
        # Create and show main window
        try:
            window = MainWindow()
            window.show()
        except Exception as e:
            logger.error(f"Error creating main window: {e}")
            QMessageBox.critical(None, "Application Error", 
                               f"Could not create main window:\n{str(e)}")
            sys.exit(1)
        
        # Run application
        try:
            sys.exit(app.exec())
        except Exception as e:
            logger.error(f"Error running application: {e}")
            sys.exit(1)
        
    except ImportError as e:
        # Handle missing dependencies gracefully
        error_msg = f"Missing required dependency: {str(e)}"
        logger.error(error_msg)
        
        # Try to show error dialog if PySide6 is available
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox
            app = QApplication(sys.argv)
            QMessageBox.critical(None, "Dependency Error", 
                               f"{error_msg}\n\nPlease install required packages:\n"
                               f"pip install PySide6 llama-cpp-python")
            sys.exit(1)
        except Exception:
            print(f"ERROR: {error_msg}")
            print("Please install required packages: pip install PySide6 llama-cpp-python")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        try:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(None, "Application Error", 
                               f"An unexpected error occurred:\n{str(e)}")
        except Exception:
            print(f"FATAL ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
