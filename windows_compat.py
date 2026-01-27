"""
Windows compatibility utilities for LocalMind
Prevents common Windows crashes and issues
"""

import platform
import sys
import os
import logging

logger = logging.getLogger(__name__)

def is_windows():
    """Check if running on Windows"""
    return platform.system() == "Windows"

def is_windows_11():
    """Check if running on Windows 11"""
    if not is_windows():
        return False
    
    try:
        # Windows 11 is version 10.0 with build >= 22000
        version = platform.version()
        build = int(version.split('.')[-1]) if '.' in version else 0
        return build >= 22000
    except:
        return False

def get_windows_safe_config():
    """Get Windows-safe configuration parameters"""
    if not is_windows():
        return {
            'n_ctx': 8192,
            'max_memory_kb': 2000,
            'use_mlock': True,
            'n_gpu_layers': -1
        }
    
    # Windows 11 can handle slightly higher settings than Windows 10
    if is_windows_11():
        return {
            'n_ctx': 4096,          # Higher context for Win11
            'max_memory_kb': 1500,  # More memory for Win11
            'use_mlock': False,     # Still disable memory locking
            'n_gpu_layers': 0       # CPU-only for stability
        }
    
    # Windows 10 and older - conservative settings
    return {
        'n_ctx': 2048,
        'max_memory_kb': 1000,
        'use_mlock': False,
        'n_gpu_layers': 0
    }

def setup_windows_environment():
    """Setup Windows-specific environment variables"""
    if is_windows():
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        os.environ['PYTHONUNBUFFERED'] = '1'
        
        # Windows 11 specific optimizations
        if is_windows_11():
            # Windows 11 has better DPI handling
            os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'
            os.environ['QT_ENABLE_HIGHDPI_SCALING'] = '1'
            os.environ['QT_SCALE_FACTOR_ROUNDING_POLICY'] = 'RoundPreferFloor'
            # Fix for Windows 11 Qt attribute issues
            os.environ['QT_LOGGING_RULES'] = 'qt.qpa.windows.debug=false'
            logger.info("Windows 11 environment configured with enhanced DPI support")
        else:
            # Windows 10 and older - disable scaling issues
            os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'
            os.environ['QT_ENABLE_HIGHDPI_SCALING'] = '0'
            logger.info("Windows 10/older environment configured for stability")
        
        logger.info("Windows environment configured for stability")

def check_windows_dependencies():
    """Check for Windows-specific dependency issues"""
    if not is_windows():
        return True
    
    try:
        import llama_cpp
        logger.info("llama-cpp-python available")
        return True
    except ImportError:
        logger.error("llama-cpp-python not properly installed for Windows")
        print("Windows Installation Fix:")
        print("pip uninstall llama-cpp-python")
        print("pip install llama-cpp-python --force-reinstall --no-cache-dir")
        return False

def get_safe_thread_count():
    """Get Windows-safe thread count"""
    cpu_count = os.cpu_count() or 4
    
    if is_windows_11():
        # Windows 11 handles threading better
        return max(1, cpu_count // 3)
    elif is_windows():
        # Windows 10 and older - more conservative
        return max(1, cpu_count // 4)
    else:
        return max(1, cpu_count // 2)

def safe_set_qt_attribute(app, attr_name, value=True):
    """Safely set Qt application attributes with Windows 11 compatibility"""
    if not is_windows():
        return True
    
    try:
        # Import Qt here to avoid circular imports
        from PySide6.QtCore import Qt
        
        # Try new-style attribute first (PySide6 6.5+)
        if hasattr(Qt.ApplicationAttribute, attr_name):
            app.setAttribute(getattr(Qt.ApplicationAttribute, attr_name), value)
            return True
        # Fallback to old-style (older PySide6)
        elif hasattr(Qt, attr_name):
            app.setAttribute(getattr(Qt, attr_name), value)
            return True
        else:
            logger.debug(f"Qt attribute {attr_name} not available")
            return False
    except (AttributeError, RuntimeError) as e:
        logger.debug(f"Could not set Qt attribute {attr_name}: {e}")
        return False
