from PySide6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, 
                               QWidget, QTextEdit, QLineEdit, QPushButton, 
                               QFileDialog, QLabel, QListWidget, 
                               QListWidgetItem, QMessageBox, QScrollArea, QMenu, QSlider, QSpinBox, QComboBox,
                               QCheckBox, QProgressDialog, QApplication, QDialog, QProgressBar)
from PySide6.QtCore import Qt, QThread, Signal, QPoint, QTimer
from PySide6.QtGui import QFont, QKeySequence, QShortcut, QAction
from llm.llama_handler import LlamaHandler, LlamaWorker, ModelLoader

# Import memory manager
from memory_manager import MemoryManager

import json
import os
import logging
import requests
from typing import Tuple
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelDownloader(QThread):
    """Cancellable model downloader using streaming API"""
    progress_updated = Signal(int, str)  # percentage, status
    download_finished = Signal(str)  # model_path
    download_failed = Signal(str)  # error_message
    
    def __init__(self, repo_id, filename, local_dir):
        super().__init__()
        self.repo_id = repo_id
        self.filename = filename
        self.local_dir = local_dir
        self._cancel_requested = False
        
    def request_cancel(self):
        """Cooperative cancel - thread will check this flag"""
        self._cancel_requested = True
        
    def run(self):
        try:
            from huggingface_hub import hf_hub_url
            import requests
            
            # Create models directory
            models_dir = Path(self.local_dir)
            models_dir.mkdir(exist_ok=True)
            
            self.progress_updated.emit(0, "Starting download...")
            
            # Get download URL
            url = hf_hub_url(self.repo_id, self.filename)
            output_path = models_dir / self.filename
            
            self.progress_updated.emit(5, "Connecting...")
            
            # Start streaming download
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            
            if total_size > 0:
                size_mb = total_size / (1024 * 1024)
                self.progress_updated.emit(10, f"Downloading {size_mb:.1f}MB...")
            else:
                self.progress_updated.emit(10, "Downloading...")
            
            # Stream download with cancel checks
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    # Check for cancel request
                    if self._cancel_requested:
                        f.close()
                        if output_path.exists():
                            output_path.unlink()  # Delete partial file
                        return  # Exit cleanly
                    
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            percent = min(95, int(downloaded / total_size * 90) + 10)
                            self.progress_updated.emit(percent, f"Downloaded {downloaded // (1024*1024)}MB...")
            
            # Final check before completion
            if self._cancel_requested:
                if output_path.exists():
                    output_path.unlink()
                return
            
            self.progress_updated.emit(100, "Download complete!")
            self.download_finished.emit(str(output_path))
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Network error: {str(e)}"
            self.download_failed.emit(error_msg)
        except Exception as e:
            error_msg = f"Download failed: {str(e)}"
            self.download_failed.emit(error_msg)


class DownloadModelDialog(QDialog):
    """Dialog for downloading models from HuggingFace"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.downloader = None
        self.downloaded_model_path = None
        self.setup_ui()
        
    def setup_ui(self):
        self.setWindowTitle("Get Local Model")
        self.setFixedSize(450, 280)
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Privacy info
        info = QLabel(
            "Models are downloaded once and run entirely on your device.\n"
            "No prompts, chats, or documents ever leave your system."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #ffffff; font-size: 13px; background-color: rgba(76, 175, 80, 0.1); padding: 10px; border-radius: 4px;")
        layout.addWidget(info)
        
        # Model selection (single option)
        model_label = QLabel("Model:")
        model_label.setStyleSheet("color: #ffffff; font-size: 14px; font-weight: bold;")
        layout.addWidget(model_label)
        
        self.model_combo = QComboBox()
        self.model_combo.setEnabled(False)  # Disable since only one option
        self.model_combo.setStyleSheet("""
            QComboBox {
                background-color: #3a3a3a;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 10px;
                border-radius: 4px;
                font-size: 12px;
            }
        """)
        
        # Add single curated model
        model_data = ("hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF", "llama-3.2-3b-instruct-q4_k_m.gguf")
        self.model_combo.addItem(
            "Llama-3.2-3B-Instruct (Q4_K_M)  •  ~2.0 GB", 
            model_data
        )
        print(f"DEBUG: Added model data: {model_data}")  # Debug
        
        layout.addWidget(self.model_combo)
        
        # Progress section (initially hidden)
        self.progress_widget = QWidget()
        progress_layout = QVBoxLayout(self.progress_widget)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        
        self.progress_label = QLabel("Ready to download")
        self.progress_label.setStyleSheet("color: #ffffff; font-size: 12px;")
        progress_layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 4px;
                text-align: center;
                background-color: #2a2a2a;
                color: #ffffff;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)
        
        layout.addWidget(self.progress_widget)
        
        layout.addStretch()
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #555555;
                color: #ffffff;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """)
        self.cancel_button.clicked.connect(self.cancel_download)
        button_layout.addWidget(self.cancel_button)
        
        self.download_button = QPushButton("Download & Load")
        self.download_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: #ffffff;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #333333;
                color: #666666;
            }
        """)
        self.download_button.clicked.connect(self.start_download)
        print("DEBUG: Connected download button to start_download")  # Debug
        button_layout.addWidget(self.download_button)
        
        layout.addLayout(button_layout)
        
        # Apply dark theme
        self.setStyleSheet("""
            QDialog {
                background-color: #2a2a2a;
                color: #ffffff;
            }
        """)
    
    def start_download(self):
        print("DEBUG: start_download called")
        
        if self.downloader and self.downloader.isRunning():
            print("DEBUG: Downloader already running")
            return
            
        # Get selected model
        current_data = self.model_combo.currentData()
        print(f"DEBUG: current_data = {current_data}")
        
        if not current_data:
            print("DEBUG: No current data, returning")
            self.progress_label.setText("No model selected")
            return
            
        repo_id, filename = current_data
        print(f"DEBUG: repo_id={repo_id}, filename={filename}")
        
        # Validate model selection
        if not repo_id or not filename:
            print("DEBUG: Invalid repo_id or filename")
            self.progress_label.setText("Invalid model selection")
            return
        
        print("DEBUG: Starting UI setup for download")
        
        # Show downloading status in main window immediately
        if self.parent():
            self.parent().model_status.setText("Downloading model…")
            self.parent().model_status.setStyleSheet("color: orange;")
        
        # Setup UI for downloading - SHOW IMMEDIATE FEEDBACK
        self.download_button.setEnabled(False)
        self.download_button.setText("Downloading...")
        self.cancel_button.setText("Cancel")
        self.cancel_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.progress_bar.setVisible(True)
        self.progress_label.setText("Downloading model (~2.0 GB). This may take a while.")
        self.progress_label.setVisible(True)
        
        # Make dialog modal during download
        self.setModal(True)
        
        # Force UI update
        self.progress_bar.repaint()
        self.progress_label.repaint()
        self.download_button.repaint()
        
        # Prevent dialog from being closed during download
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)
        
        # Create models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        print(f"DEBUG: Created models directory: {models_dir}")
        
        # Start download
        print("DEBUG: Creating ModelDownloader")
        self.downloader = ModelDownloader(repo_id, filename, str(models_dir))
        
        # Store strong reference in parent window
        if hasattr(self.parent(), 'download_worker'):
            self.parent().download_worker = self.downloader
        
        print("DEBUG: Connecting signals")
        self.downloader.progress_updated.connect(self.update_progress)
        self.downloader.download_finished.connect(self.download_completed)
        self.downloader.download_failed.connect(self.download_error)
        self.downloader.finished.connect(self.downloader.deleteLater)  # Clean shutdown
        print("DEBUG: Signals connected")
        
        print("DEBUG: Starting downloader thread")
        self.downloader.start()
        print("DEBUG: Downloader thread started")
    
    def simulate_progress(self):
        """Simulate progress updates during download"""
        print("DEBUG: simulate_progress called")
        if self.downloader and self.downloader.isRunning():
            if self.current_progress < 90:
                self.current_progress += 10
                self.progress_bar.setValue(self.current_progress)
                self.progress_label.setText(f"Downloading model... {self.current_progress}%")
                print(f"DEBUG: Updated progress to {self.current_progress}%")
        else:
            print("DEBUG: Downloader not running, stopping timer")
            if hasattr(self, 'progress_timer'):
                self.progress_timer.stop()
    
    def update_progress(self, percentage, status):
        print(f"DEBUG: update_progress called: {percentage}%, {status}")
        self.progress_bar.setValue(percentage)
        self.progress_label.setText(status)
    
    def download_completed(self, model_path):
        print(f"DEBUG: download_completed called with path: {model_path}")
        
        self.downloaded_model_path = model_path
        self.progress_bar.setValue(100)
        self.progress_label.setText("Model ready")
        self.download_button.setText("Close")
        self.download_button.setEnabled(True)
        self.download_button.clicked.disconnect()
        self.download_button.clicked.connect(self.accept)  # Safe to accept now
        
        # Force UI update
        self.progress_bar.repaint()
        self.progress_label.repaint()
        self.download_button.repaint()
        
        # Auto-load the model in parent window
        if self.parent():
            parent = self.parent()
            parent.model_path = model_path
            parent.load_button.setEnabled(True)
            parent.save_settings()
            parent.load_model()
            model_name = os.path.basename(model_path).replace('.gguf', '')
            parent.model_status.setText(f"{model_name}\nStatus: Loaded")
            parent.model_status.setStyleSheet("")  # Reset color
            
            # Clean up worker reference
            parent.download_worker = None
    
    def download_error(self, error_message):
        print(f"DEBUG: download_error called: {error_message}")
        
        self.progress_bar.setValue(0)
        self.progress_label.setText(f"Download failed: {error_message}")
        self.download_button.setText("Retry")
        self.download_button.setEnabled(True)
        
        # Re-enable close button
        self.setWindowFlags(self.windowFlags() | Qt.WindowCloseButtonHint)
        
        # Force UI update
        self.progress_bar.repaint()
        self.progress_label.repaint()
        self.download_button.repaint()
    
    def cancel_download(self):
        print("DEBUG: cancel_download called")
        if self.downloader and self.downloader.isRunning():
            print("DEBUG: Cancelling running downloader")
            self.downloader.cancel()
            self.progress_label.setText("Cancelling download...")
        else:
            print("DEBUG: No running downloader to cancel")
        self.reject()
    
    def accept(self):
        print("DEBUG: Dialog.accept() called")
        super().accept()
    
    def reject(self):
        print("DEBUG: Dialog.reject() called")
        super().reject()
    
    def closeEvent(self, event):
        """Handle dialog close event"""
        print("DEBUG: Dialog closeEvent called")
        if self.downloader and self.downloader.isRunning():
            QMessageBox.information(
                self,
                "Download in progress",
                "Please cancel the download before closing."
            )
            event.ignore()
            return
        event.accept()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.llama_handler = None
        self.model_loader = None
        self.chats = {}
        self.chat_workers = {}
        self.current_chat_id = None
        self.chat_counter = 0
        self.chats_dir = "chats"
        self.settings_file = "settings.json"
        self.model_path = None
        self.drag_position = QPoint()
        
        # Strong references for download worker
        self.download_worker = None
        
        # Default parameters
        self.temperature = 0.70
        self.max_tokens = 512
        
        # Chat profiles
        self.profiles = {
            "general": {
                "name": "General",
                "system_prompt": "You are a helpful assistant. Answer questions clearly and completely. Do not ask follow-up questions.",
                "temperature": 0.70,
                "max_tokens": 512
            },
            "document": {
                "name": "Document",
                "system_prompt": "You are a document analysis assistant. Answer questions based on the learned knowledge provided in your context. Be precise and reference specific concepts, facts, or definitions from the learned material. Do not ask follow-up questions.",
                "temperature": 0.30,
                "max_tokens": 800
            },
            "student": {
                "name": "Student",
                "system_prompt": "You are a helpful tutor. Explain concepts clearly and encourage learning with examples. Do not ask follow-up questions.",
                "temperature": 0.60,
                "max_tokens": 800
            },
            "code": {
                "name": "Code",
                "system_prompt": "You are a programming assistant. Provide clean, well-commented code with explanations. Do not ask follow-up questions.",
                "temperature": 0.30,
                "max_tokens": 1024
            },
            "writer": {
                "name": "Writer",
                "system_prompt": "You are a creative writing assistant. Help with storytelling, editing, and creative expression. Do not ask follow-up questions.",
                "temperature": 0.80,
                "max_tokens": 1200
            }
        }
        
        # Memory management
        self.memory_manager = MemoryManager()
        
        # Download button styles
        self.IDLE_STYLE = """
            QPushButton {
                background-color: #1e1e1e;
                color: #9be9a8;
                border: 1px solid #2f855a;
                border-radius: 6px;
                padding: 8px;
            }
        """
        
        self.DOWNLOADING_STYLE = """
            QPushButton {{
                color: black;
                font-weight: bold;
                border-radius: 6px;
                padding: 8px;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #38a169,
                    stop:{progress} #38a169,
                    stop:{progress} #ecc94b,
                    stop:1 #ecc94b
                );
            }}
        """
        
        self.DONE_STYLE = """
            QPushButton {
                background-color: #38a169;
                color: black;
                font-weight: bold;
                border-radius: 6px;
                padding: 8px;
            }
        """
        
        os.makedirs(self.chats_dir, exist_ok=True)
        self.load_settings()
        self.setup_ui()
        self.apply_dark_theme()
        self.load_existing_chats()
        # Always create a new chat on startup
        self.new_chat()
        
        # Initialize document list
        self.refresh_document_list()
        
        # Update UI based on saved model path
        if self.model_path:
            self.model_status.setText(f"Selected: {os.path.basename(self.model_path)}")
            self.load_button.setEnabled(True)
        
        # Initialize default temperature preset
        self.update_preset_selection(0.7)
        
    def setup_ui(self):
        self.setWindowTitle("LocalMind")
        self.setGeometry(100, 100, 1200, 700)
        
        # Central widget with main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main vertical layout for the entire window
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Content area with horizontal layout
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # Left sidebar (fixed width)
        sidebar = self.create_sidebar()
        content_layout.addWidget(sidebar)
        
        # Main chat area (expandable)
        chat_area = self.create_chat_area()
        content_layout.addWidget(chat_area, 1)  # stretch factor 1 makes it expandable
        
        main_layout.addWidget(content_widget)
        
        # Add keyboard shortcuts
        self.setup_shortcuts()
        
    def create_sidebar(self):
        sidebar = QWidget()
        sidebar.setFixedWidth(280)
        sidebar.setObjectName("sidebar")
        
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Sessions section (moved to top)
        sessions_label = QLabel("Sessions")
        sessions_label.setObjectName("section_label")
        layout.addWidget(sessions_label)
        
        # Session controls
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)
        
        self.new_chat_button = QPushButton("New")
        self.new_chat_button.clicked.connect(self.new_chat)
        
        self.delete_chat_button = QPushButton("Delete")
        self.delete_chat_button.clicked.connect(self.delete_current_chat)
        self.delete_chat_button.setEnabled(False)
        self.delete_chat_button.setObjectName("delete_button")
        
        controls_layout.addWidget(self.new_chat_button)
        controls_layout.addWidget(self.delete_chat_button)
        layout.addLayout(controls_layout)
        
        # Chat list
        self.chat_list = QListWidget()
        self.chat_list.itemClicked.connect(self.switch_chat)
        self.chat_list.setObjectName("chat_list")
        self.chat_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.chat_list.customContextMenuRequested.connect(self.show_chat_context_menu)
        layout.addWidget(self.chat_list)
        
        layout.addSpacing(20)
        
        # Model section (moved to middle)
        model_label = QLabel("Model")
        model_label.setObjectName("section_label")
        layout.addWidget(model_label)
        
        self.model_status = QLabel("No model loaded")
        self.model_status.setObjectName("model_status")
        layout.addWidget(self.model_status)
        
        self.browse_button = QPushButton("Browse Model")
        self.browse_button.clicked.connect(self.browse_model)
        layout.addWidget(self.browse_button)
        
        self.load_button = QPushButton("Load Model")
        self.load_button.clicked.connect(self.load_model)
        self.load_button.setEnabled(False)
        layout.addWidget(self.load_button)
        
        # Download model button with cancel support
        self.download_row = QHBoxLayout()
        self.download_row.setSpacing(5)
        
        self.download_btn = QPushButton("Download Model")
        self.download_btn.clicked.connect(self.start_model_download)
        self.download_btn.setStyleSheet(self.IDLE_STYLE)
        
        self.cancel_btn = QPushButton("✖")
        self.cancel_btn.setFixedWidth(32)
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self.cancel_download)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        
        self.download_row.addWidget(self.download_btn)
        self.download_row.addWidget(self.cancel_btn)
        layout.addLayout(self.download_row)
        
        layout.addSpacing(20)
        
        # Parameters section
        params_label = QLabel("Parameters")
        params_label.setObjectName("section_label")
        layout.addWidget(params_label)
        
        # Profile indicator
        profile_info = QLabel("(Set by Profile)")
        profile_info.setObjectName("profile_info")
        profile_info.setAlignment(Qt.AlignCenter)
        layout.addWidget(profile_info)
        
        # Temperature control
        temp_layout = QVBoxLayout()
        temp_layout.setSpacing(5)
        
        temp_header = QHBoxLayout()
        temp_label = QLabel("Temperature:")
        temp_label.setObjectName("param_label")
        
        # Add info icon with tooltip
        info_icon = QLabel("!")
        info_icon.setStyleSheet("color: #888; font-weight: bold; font-size: 12px; border: 1px solid #888; border-radius: 8px; width: 16px; height: 16px; text-align: center;")
        info_icon.setFixedSize(16, 16)
        info_icon.setAlignment(Qt.AlignCenter)
        info_icon.setToolTip("Temperature controls randomness. Lower (Robotic) = more precise and factual. Higher (Creative) = more creative and varied")
        info_icon.mousePressEvent = lambda event: QMessageBox.information(self, "Temperature Info", "Temperature controls randomness. Lower (Robotic) = more precise and factual. Higher (Creative) = more creative and varied")
        
        self.temp_value = QLabel("0.70")
        self.temp_value.setObjectName("param_value")
        temp_header.addWidget(temp_label)
        temp_header.addWidget(info_icon)
        temp_header.addStretch()
        temp_header.addWidget(self.temp_value)
        temp_layout.addLayout(temp_header)
        
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setMinimum(1)
        self.temp_slider.setMaximum(200)
        self.temp_slider.setValue(70)
        self.temp_slider.setObjectName("param_slider")
        self.temp_slider.valueChanged.connect(self.update_temperature)
        temp_layout.addWidget(self.temp_slider)
        
        # Temperature preset buttons
        preset_layout = QHBoxLayout()
        preset_layout.setSpacing(5)
        
        self.robotic_btn = QPushButton("Robotic")
        self.robotic_btn.setObjectName("preset_button")
        self.robotic_btn.setStyleSheet("font-size: 10px;")
        self.robotic_btn.clicked.connect(lambda: self.set_temperature_preset(0.2))
        
        self.default_btn = QPushButton("Default")
        self.default_btn.setObjectName("preset_button")
        self.default_btn.setStyleSheet("font-size: 10px;")
        self.default_btn.clicked.connect(lambda: self.set_temperature_preset(0.7))
        
        self.creative_btn = QPushButton("Creative")
        self.creative_btn.setObjectName("preset_button")
        self.creative_btn.setStyleSheet("font-size: 10px;")
        self.creative_btn.clicked.connect(lambda: self.set_temperature_preset(1.0))
        
        preset_layout.addWidget(self.robotic_btn)
        preset_layout.addWidget(self.default_btn)
        preset_layout.addWidget(self.creative_btn)
        temp_layout.addLayout(preset_layout)
        
        layout.addLayout(temp_layout)
        layout.addSpacing(10)
        
        # Token control
        token_layout = QVBoxLayout()
        token_layout.setSpacing(5)
        
        token_header = QHBoxLayout()
        token_label = QLabel("Max Tokens:")
        token_label.setObjectName("param_label")
        token_header.addWidget(token_label)
        token_header.addStretch()
        
        self.token_spinbox = QSpinBox()
        self.token_spinbox.setMinimum(1)
        self.token_spinbox.setMaximum(4096)
        self.token_spinbox.setValue(512)
        self.token_spinbox.setObjectName("param_spinbox")
        self.token_spinbox.valueChanged.connect(self.update_tokens)
        token_header.addWidget(self.token_spinbox)
        
        token_layout.addLayout(token_header)
        layout.addLayout(token_layout)
        
        layout.addSpacing(20)
        
        # Documents section
        documents_label = QLabel("📄 DOCUMENTS")
        documents_label.setObjectName("section_label")
        layout.addWidget(documents_label)
        
        # Document controls
        doc_controls_layout = QHBoxLayout()
        doc_controls_layout.setSpacing(10)
        
        self.import_doc_button = QPushButton("Import Document")
        self.import_doc_button.clicked.connect(self.import_document)
        doc_controls_layout.addWidget(self.import_doc_button)
        
        layout.addLayout(doc_controls_layout)
        
        # Document list
        self.document_list = QListWidget()
        self.document_list.setObjectName("document_list")
        self.document_list.setMaximumHeight(120)
        self.document_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.document_list.customContextMenuRequested.connect(self.show_document_context_menu)
        self.document_list.itemClicked.connect(self.on_document_selected)
        layout.addWidget(self.document_list)
        
        return sidebar
        
    def create_chat_area(self):
        chat_area = QWidget()
        chat_area.setObjectName("chat_area")
        
        # Main vertical layout for chat area
        layout = QVBoxLayout(chat_area)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Profile selection area
        profile_container = QWidget()
        profile_container.setObjectName("profile_container")
        profile_layout = QHBoxLayout(profile_container)
        profile_layout.setContentsMargins(20, 10, 20, 5)
        profile_layout.setSpacing(10)
        
        profile_label = QLabel("Profile:")
        profile_label.setObjectName("profile_label")
        profile_layout.addWidget(profile_label)
        
        self.profile_combo = QComboBox()
        self.profile_combo.setObjectName("profile_combo")
        for profile_id, profile in self.profiles.items():
            self.profile_combo.addItem(profile["name"], profile_id)
        self.profile_combo.currentTextChanged.connect(self.on_profile_changed)
        profile_layout.addWidget(self.profile_combo)
        
        profile_layout.addStretch()
        
        # Export Chat button
        self.export_button = QPushButton("Export Chat")
        self.export_button.setObjectName("export_button")
        self.export_button.clicked.connect(self.export_current_chat)
        self.export_button.setFixedSize(100, 30)
        profile_layout.addWidget(self.export_button)
        layout.addWidget(profile_container)
        
        # Active document indicator
        self.active_doc_container = QWidget()
        self.active_doc_container.setObjectName("active_doc_container")
        self.active_doc_container.setVisible(False)  # Hidden by default
        active_doc_layout = QHBoxLayout(self.active_doc_container)
        active_doc_layout.setContentsMargins(20, 10, 20, 10)
        active_doc_layout.setSpacing(10)
        
        self.active_doc_label = QLabel()
        self.active_doc_label.setObjectName("active_doc_label")
        active_doc_layout.addWidget(self.active_doc_label)
        
        active_doc_layout.addStretch()
        
        self.change_doc_button = QPushButton("Change")
        self.change_doc_button.setObjectName("change_doc_button")
        self.change_doc_button.clicked.connect(self.show_document_selection)
        self.change_doc_button.setFixedSize(95, 32)
        active_doc_layout.addWidget(self.change_doc_button)
        
        self.clear_doc_button = QPushButton("Clear")
        self.clear_doc_button.setObjectName("clear_doc_button")
        self.clear_doc_button.clicked.connect(self.clear_active_document)
        self.clear_doc_button.setFixedSize(85, 32)
        active_doc_layout.addWidget(self.clear_doc_button)
        
        layout.addWidget(self.active_doc_container)
        
        # System prompt area
        system_prompt_container = QWidget()
        system_prompt_container.setObjectName("system_prompt_container")
        system_layout = QVBoxLayout(system_prompt_container)
        system_layout.setContentsMargins(20, 15, 20, 10)
        system_layout.setSpacing(5)
        
        system_label = QLabel("System Prompt:")
        system_label.setObjectName("system_label")
        system_layout.addWidget(system_label)
        
        self.system_prompt = QTextEdit()
        self.system_prompt.setPlaceholderText("Enter system prompt (optional)...")
        self.system_prompt.setPlainText("You are a helpful assistant")
        self.system_prompt.setFixedHeight(60)
        self.system_prompt.setObjectName("system_prompt")
        system_layout.addWidget(self.system_prompt)
        
        layout.addWidget(system_prompt_container)
        
        # Separator line
        separator = QWidget()
        separator.setFixedHeight(1)
        separator.setObjectName("separator")
        layout.addWidget(separator)
        
        # Messages area (expandable)
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setObjectName("chat_display")
        
        # Fix QTextDocument formatting - preserve block spacing
        doc = self.chat_display.document()
        doc.setDocumentMargin(10)
        
        # Set default text format with proper line spacing
        from PySide6.QtGui import QTextBlockFormat, QTextCursor
        cursor = QTextCursor(doc)
        block_format = QTextBlockFormat()
        block_format.setLineHeight(120, 1)  # 1 = ProportionalHeight
        block_format.setBottomMargin(8)
        cursor.setBlockFormat(block_format)
        
        layout.addWidget(self.chat_display, 1)  # stretch factor 1 makes it expandable
        
        # Input bar (fixed height)
        input_bar = self.create_input_bar()
        layout.addWidget(input_bar)
        
        return chat_area
        
    def create_input_bar(self):
        input_bar = QWidget()
        input_bar.setFixedHeight(80)
        input_bar.setObjectName("input_bar")
        
        layout = QHBoxLayout(input_bar)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(15)
        
        # Input field (expandable)
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your message...")
        self.input_field.returnPressed.connect(self.send_message)
        self.input_field.setObjectName("input_field")
        layout.addWidget(self.input_field, 1)  # stretch factor 1 makes it expandable
        
        # Buttons (fixed size)
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        self.send_button.setEnabled(False)
        self.send_button.setFixedSize(80, 40)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_generation)
        self.stop_button.setEnabled(False)
        self.stop_button.setFixedSize(80, 40)
        
        layout.addWidget(self.send_button)
        layout.addWidget(self.stop_button)
        
        return input_bar
        
    def apply_dark_theme(self):
        self.setStyleSheet("""
            * {
                font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
            }
            
            QMainWindow {
                background-color: #0f1115;
                color: #f0f2f5;
            }
            
            #sidebar {
                background-color: #16181d;
                border-right: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 0px;
            }
            
            #section_label {
                color: #f0f2f5;
                font-weight: 600;
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 1.2px;
                padding: 16px 20px 8px 20px;
                background-color: transparent;
                border-radius: 0px;
                border: none;
            }
            
            #model_status {
                color: #5bb885;
                font-size: 11px;
                padding: 12px 20px;
                background-color: rgba(91, 184, 133, 0.08);
                border-radius: 10px;
                border: 1px solid rgba(91, 184, 133, 0.15);
            }
            
            #chat_area {
                background-color: #0f1115;
                border-radius: 0px;
            }
            
            #chat_display {
                background-color: #0f1115;
                border: none;
                color: #f0f2f5;
                padding: 32px;
                font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
                font-size: 14px;
                line-height: 1.6;
                border-radius: 0px;
            }
            
            #input_bar {
                background-color: #0f1115;
                border-top: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 0px;
                padding: 20px;
            }
            
            #input_field {
                background-color: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.15);
                color: #f0f2f5;
                padding: 14px 18px;
                border-radius: 12px;
                font-size: 14px;
            }
            
            #input_field:focus {
                border-color: rgba(91, 184, 133, 0.4);
                background-color: rgba(255, 255, 255, 0.06);
            }
            
            #system_prompt_container {
                background-color: rgba(255, 255, 255, 0.03);
                border-radius: 12px;
                padding: 16px;
                border: 1px solid rgba(255, 255, 255, 0.06);
            }
            
            #system_label {
                color: #5bb885;
                font-size: 11px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.8px;
                margin-bottom: 8px;
            }
            
            #system_prompt {
                background-color: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.15);
                color: #f0f2f5;
                padding: 12px 14px;
                border-radius: 8px;
                font-size: 13px;
            }
            
            #system_prompt:focus {
                border-color: rgba(91, 184, 133, 0.3);
                background-color: rgba(255, 255, 255, 0.06);
            }
            
            #separator {
                background-color: rgba(255, 255, 255, 0.06);
            }
            
            #chat_list {
                background-color: transparent;
                border: none;
                color: #f0f2f5;
                border-radius: 8px;
                padding: 4px;
            }
            
            #chat_list::item {
                padding: 12px 16px;
                border-radius: 8px;
                margin: 2px 0;
                background-color: transparent;
            }
            
            #chat_list::item:selected {
                background-color: rgba(91, 184, 133, 0.15);
                border: 1px solid rgba(91, 184, 133, 0.25);
            }
            
            #chat_list::item:hover {
                background-color: rgba(255, 255, 255, 0.05);
            }
            
            QPushButton {
                background-color: transparent;
                border: 1px solid rgba(91, 184, 133, 0.4);
                color: #5bb885;
                padding: 10px 20px;
                border-radius: 8px;
                font-weight: 500;
                font-size: 13px;
            }
            
            QPushButton:hover {
                background-color: rgba(91, 184, 133, 0.12);
                border-color: rgba(91, 184, 133, 0.6);
            }
            
            QPushButton:pressed {
                background-color: rgba(91, 184, 133, 0.2);
            }
            
            QPushButton:disabled {
                background-color: transparent;
                border-color: rgba(255, 255, 255, 0.1);
                color: #6b7280;
            }
            
            #delete_button {
                border-color: rgba(239, 68, 68, 0.4);
                color: #ef4444;
            }
            
            #delete_button:hover {
                background-color: rgba(239, 68, 68, 0.12);
                border-color: rgba(239, 68, 68, 0.6);
            }
            
            #export_button {
                background-color: transparent;
                border: 1px solid rgba(91, 184, 133, 0.3);
                color: #5bb885;
                font-size: 11px;
                padding: 6px 12px;
            }
            
            #export_button:hover {
                background-color: rgba(91, 184, 133, 0.1);
            }
            
            #download_model_button {
                background-color: transparent;
                border: 1px solid rgba(100, 149, 237, 0.3);
                color: #6495ed;
                font-size: 11px;
                padding: 6px 12px;
            }
            
            #download_model_button:hover {
                background-color: rgba(100, 149, 237, 0.1);
            }
            
            #param_label {
                color: #9ca3af;
                font-size: 11px;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            #param_value {
                color: #5bb885;
                font-size: 12px;
                font-weight: 600;
                background-color: rgba(91, 184, 133, 0.08);
                padding: 6px 10px;
                border-radius: 6px;
                border: 1px solid rgba(91, 184, 133, 0.15);
            }
            
            #param_slider {
                background-color: transparent;
            }
            
            #param_slider::groove:horizontal {
                border: 1px solid rgba(255, 255, 255, 0.1);
                height: 4px;
                background-color: rgba(255, 255, 255, 0.05);
                border-radius: 2px;
            }
            
            #param_slider::handle:horizontal {
                background-color: #5bb885;
                border: 2px solid rgba(91, 184, 133, 0.3);
                width: 14px;
                height: 14px;
                border-radius: 7px;
                margin: -6px 0;
            }
            
            #param_slider::handle:horizontal:hover {
                background-color: #5bb885;
                border-color: rgba(91, 184, 133, 0.5);
            }
            
            #param_spinbox {
                background-color: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.15);
                color: #5bb885;
                padding: 6px 10px;
                border-radius: 6px;
                font-weight: 600;
                font-size: 12px;
            }
            
            #param_spinbox:focus {
                border-color: rgba(91, 184, 133, 0.3);
            }
            
            #profile_container {
                background-color: rgba(255, 255, 255, 0.03);
                border-radius: 12px;
                padding: 16px;
                border: 1px solid rgba(255, 255, 255, 0.06);
            }
            
            #profile_label {
                color: #5bb885;
                font-size: 11px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.8px;
            }
            
            #profile_combo {
                background-color: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.15);
                color: #f0f2f5;
                padding: 8px 14px;
                border-radius: 8px;
                font-size: 12px;
                min-width: 100px;
            }
            
            #profile_combo:focus {
                border-color: rgba(91, 184, 133, 0.3);
            }
            
            #profile_combo::drop-down {
                border: none;
                background-color: rgba(91, 184, 133, 0.2);
                border-radius: 4px;
            }
            
            #profile_combo::down-arrow {
                image: none;
                border: none;
                width: 0;
                height: 0;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #f0f2f5;
            }
            
            #profile_combo QAbstractItemView {
                background-color: #252932;
                border: 1px solid rgba(255, 255, 255, 0.1);
                color: #f0f2f5;
                selection-background-color: rgba(91, 184, 133, 0.2);
            }
            
            #profile_info {
                color: #9ca3af;
                font-size: 10px;
                font-style: italic;
                padding: 4px 0;
            }
            
            #document_list {
                background-color: transparent;
                border: none;
                color: #f0f2f5;
                border-radius: 8px;
                padding: 4px;
            }
            
            #document_list::item {
                padding: 10px 14px;
                border-radius: 8px;
                margin: 2px 0;
                background-color: transparent;
                font-size: 12px;
            }
            
            #document_list::item:hover {
                background-color: rgba(255, 255, 255, 0.05);
            }
            
            #document_list::item:selected {
                background-color: rgba(59, 130, 246, 0.15);
                border: 1px solid rgba(59, 130, 246, 0.25);
            }
            
            #active_doc_container {
                background-color: rgba(59, 130, 246, 0.1);
                border: 1px solid rgba(59, 130, 246, 0.2);
                border-radius: 10px;
                padding: 12px;
            }
            
            #active_doc_label {
                color: #f0f2f5;
                font-size: 12px;
                font-weight: 600;
            }
            
            #change_doc_button, #clear_doc_button {
                background-color: transparent;
                color: #3b82f6;
                border: 1px solid rgba(59, 130, 246, 0.3);
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                text-align: center;
            }
            
            #change_doc_button:hover, #clear_doc_button:hover {
                background-color: rgba(59, 130, 246, 0.1);
            }
            
            #change_doc_button:hover, #clear_doc_button:hover {
                background-color: #4a7bc8;
            }
            
            QCheckBox {
                color: #ffffff;
                spacing: 5px;
            }
            
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 2px solid #228B22;
                background-color: transparent;
            }
            
            QCheckBox::indicator:checked {
                background-color: #228B22;
                border: 2px solid #32CD32;
            }
            
            QCheckBox::indicator:hover {
                border: 2px solid #32CD32;
            }
        """)
    
    def load_settings(self):
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    self.model_path = settings.get('model_path')
        except FileNotFoundError:
            logger.info("Settings file not found, using defaults")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in settings file: {e}")
            QMessageBox.warning(self, "Settings Error", "Settings file is corrupted. Using default settings.")
        except PermissionError:
            logger.error("Permission denied reading settings file")
            QMessageBox.warning(self, "Settings Error", "Permission denied reading settings file.")
        except Exception as e:
            logger.error(f"Unexpected error loading settings: {e}")
            QMessageBox.warning(self, "Settings Error", f"Could not load settings: {e}")
    
    def save_settings(self):
        try:
            settings = {'model_path': self.model_path}
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
        except PermissionError:
            logger.error("Permission denied writing settings file")
            QMessageBox.warning(self, "Settings Error", "Permission denied saving settings.")
        except OSError as e:
            logger.error(f"OS error saving settings: {e}")
            QMessageBox.warning(self, "Settings Error", f"Could not save settings: {e}")
        except Exception as e:
            logger.error(f"Unexpected error saving settings: {e}")
            QMessageBox.warning(self, "Settings Error", f"Could not save settings: {e}")
    
    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Browse GGUF Model", "", "GGUF Files (*.gguf)")
        
        if file_path:
            # Validate file exists
            if not os.path.exists(file_path):
                QMessageBox.warning(self, "File Error", "Selected model file does not exist.")
                return
                
            # Validate file extension
            if not file_path.lower().endswith('.gguf'):
                QMessageBox.warning(self, "File Error", "Please select a valid GGUF model file.")
                return
                
            # Validate file permissions
            if not os.access(file_path, os.R_OK):
                QMessageBox.warning(self, "Permission Error", "Cannot read the selected model file. Check file permissions.")
                return
                
            # Validate file size (reasonable limit: 50GB)
            try:
                file_size = os.path.getsize(file_path)
                if file_size > 50 * 1024 * 1024 * 1024:  # 50GB
                    QMessageBox.warning(self, "File Too Large", "Model file exceeds the 50GB size limit.")
                    return
                if file_size == 0:
                    QMessageBox.warning(self, "Invalid File", "Model file is empty.")
                    return
            except OSError as e:
                QMessageBox.warning(self, "File Error", f"Cannot access file: {e}")
                return
                
            self.model_path = file_path
            self.model_status.setText(f"Selected: {os.path.basename(file_path)}")
            self.load_button.setEnabled(True)
            self.save_settings()
        
    def load_model(self):
        if not self.model_path:
            return
        
        # Create progress dialog
        self.model_progress = QProgressDialog("Loading model...", None, 0, 0, self)
        self.model_progress.setWindowTitle("LocalMind")
        self.model_progress.setWindowModality(Qt.WindowModal)
        self.model_progress.setCancelButton(None)
        self.model_progress.setMinimumDuration(0)
        self.model_progress.show()
        
        # Disable load button during loading
        self.load_button.setEnabled(False)
        
        # Create and start model loader thread
        self.model_loader = ModelLoader(self.model_path)
        self.model_loader.model_loaded.connect(self.on_model_loaded)
        self.model_loader.start()
    
    def on_model_loaded(self, success: bool, message: str):
        """Handle model loading completion"""
        # Close progress dialog
        if hasattr(self, 'model_progress'):
            self.model_progress.close()
        
        # Re-enable load button
        self.load_button.setEnabled(True)
        
        if success:
            self.llama_handler = self.model_loader.llama_handler
            self.memory_manager.set_llm_handler(self.llama_handler)
            self.model_status.setText(f"● Loaded: {os.path.basename(self.model_path)}")
            self.model_status.setStyleSheet("""
                color: #4ade80;
                font-size: 11px;
                padding: 12px 20px;
                background-color: rgba(74, 222, 128, 0.12);
                border-radius: 10px;
                border: 1px solid rgba(74, 222, 128, 0.3);
                font-weight: 600;
            """)
            self.chat_display.append("✓ Model loaded successfully!\n")
            self.update_ui_state()
        else:
            self.chat_display.append(f"✗ Failed to load model: {message}\n")
            QMessageBox.critical(self, "Model Loading Error", message)
                
    def send_message(self):
        if not self.llama_handler or not self.input_field.text().strip() or not self.current_chat_id:
            return
        
        # Safety check
        if self.current_chat_id not in self.chats:
            return
            
        # Check if this chat is already generating
        if self.chats[self.current_chat_id].get("is_generating", False):
            return
        
        # Clean up any existing worker for this chat
        if self.current_chat_id in self.chat_workers:
            old_worker = self.chat_workers[self.current_chat_id]
            if old_worker.isRunning():
                old_worker.stop()
                old_worker.wait(1000)
            del self.chat_workers[self.current_chat_id]
            
        # Capture chat ID to prevent race conditions
        target_chat_id = self.current_chat_id
        user_message = self.input_field.text().strip()
        self.input_field.clear()
        
        # Clear draft message since it's being sent
        self.chats[target_chat_id]["draft_message"] = ""
        
        # Update chat name with first message
        self.update_chat_name(user_message)
        
        # Store user message in chat data
        self.chats[target_chat_id]["messages"].append({
            "role": "user",
            "content": user_message
        })
        
        # Add message to current chat display with proper formatting
        from PySide6.QtGui import QTextBlockFormat, QTextCursor
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        block_format = QTextBlockFormat()
        block_format.setLineHeight(120, 1)  # 1 = ProportionalHeight
        block_format.setBottomMargin(8)
        cursor.setBlockFormat(block_format)
        
        cursor.insertHtml(f"<b>You:</b> {user_message}<br><br>")
        cursor.insertHtml("<b>AI:</b> ")
        
        # Build prompt with system prompt if provided
        system_prompt = self.system_prompt.toPlainText().strip()
        
        # Get memory context (all learned knowledge)
        memory_context = self.memory_manager.get_memory_context()
        
        # Check if asking about documents but no memory
        if not memory_context and any(word in user_message.lower() for word in ['chapter', 'document', 'summary', 'book', 'pdf']):
            # Enhance system prompt to tell user no documents loaded
            if system_prompt:
                system_prompt = f"{system_prompt}\n\nIMPORTANT: No documents have been imported yet. If the user asks about documents, chapters, or summaries, politely inform them that no documents are available and they need to import documents first using the 'Import Document' button."
            else:
                system_prompt = "You are a helpful assistant. IMPORTANT: No documents have been imported yet. If the user asks about documents, chapters, or summaries, politely inform them that no documents are available and they need to import documents first using the 'Import Document' button."
        
        # Mark this chat as generating
        self.chats[target_chat_id]["is_generating"] = True
        
        # Get profile settings for this chat
        profile_id = self.chats[target_chat_id].get("profile", "general")
        profile = self.profiles.get(profile_id, self.profiles["general"])
        
        # Initialize assistant response tracking for this chat
        if target_chat_id not in self.chats:
            return
        self.chats[target_chat_id]["current_response"] = ""
        
        worker = LlamaWorker(
            self.llama_handler, 
            user_message,  # Just the user message, not pre-formatted
            system_prompt,
            memory_context,
            profile["temperature"], 
            profile["max_tokens"]
        )
        # Use captured chat_id to prevent race conditions
        worker.token_received.connect(lambda token: self.append_token(target_chat_id, token))
        worker.finished.connect(lambda: self.generation_finished(target_chat_id))
        worker.start()
        
        # Store worker reference
        self.chat_workers[target_chat_id] = worker
        
        # Update UI
        self.update_ui_state()
        
    def append_token(self, chat_id, token):
        """Only append token if we're viewing the chat that's generating"""
        if chat_id in self.chats:
            # Collect assistant response
            if "current_response" not in self.chats[chat_id]:
                self.chats[chat_id]["current_response"] = ""
            self.chats[chat_id]["current_response"] += token
            
            # Update display if we're viewing this chat
            if chat_id == self.current_chat_id:
                # Move cursor to end before inserting to prevent broken text
                cursor = self.chat_display.textCursor()
                cursor.movePosition(cursor.MoveOperation.End)
                self.chat_display.setTextCursor(cursor)
                self.chat_display.insertPlainText(token)
                self.chat_display.ensureCursorVisible()
            
    def generation_finished(self, chat_id):
        """Handle generation completion for specific chat"""
        # Safety check
        if chat_id not in self.chats:
            return
            
        # Store assistant response in messages
        if "current_response" in self.chats[chat_id]:
            assistant_response = self.chats[chat_id]["current_response"].strip()
            if assistant_response:
                self.chats[chat_id]["messages"].append({
                    "role": "assistant",
                    "content": assistant_response
                })
            # Clean up temporary response tracking
            del self.chats[chat_id]["current_response"]
        
        # Mark chat as not generating
        self.chats[chat_id]["is_generating"] = False
        
        # Remove worker reference
        if chat_id in self.chat_workers:
            del self.chat_workers[chat_id]
        
        # Update display if we're viewing this chat
        if chat_id == self.current_chat_id:
            self.chat_display.append("\n\n")
            self.update_ui_state()
        
        # Save the chat with new messages
        self.save_chat_by_id(chat_id)
        
    def stop_generation(self):
        """Stop generation for current chat only"""
        if not self.current_chat_id or self.current_chat_id not in self.chat_workers:
            return
            
        try:
            worker = self.chat_workers[self.current_chat_id]
            if worker.isRunning():
                # Request stop
                worker.stop()
                
                # Wait for graceful shutdown
                if not worker.wait(2000):  # Wait max 2 seconds
                    logger.warning("Worker did not stop gracefully, terminating")
                    worker.terminate()
                    worker.wait(1000)
                    
                self.chat_display.append("\n[Generation stopped]\n\n")
            
            # Disconnect signals to prevent stray emissions
            try:
                worker.token_received.disconnect()
                worker.finished.disconnect()
            except:
                pass
                
            # Store partial assistant response if any
            if self.current_chat_id in self.chats and "current_response" in self.chats[self.current_chat_id]:
                assistant_response = self.chats[self.current_chat_id]["current_response"].strip()
                if assistant_response:
                    self.chats[self.current_chat_id]["messages"].append({
                        "role": "assistant",
                        "content": assistant_response
                    })
                # Clean up temporary response tracking
                del self.chats[self.current_chat_id]["current_response"]
                
            # Mark as not generating
            self.chats[self.current_chat_id]["is_generating"] = False
            
            # Clean up worker
            if self.current_chat_id in self.chat_workers:
                del self.chat_workers[self.current_chat_id]
                
        except Exception as e:
            logger.error(f"Error stopping generation: {e}")
            # Force cleanup on error
            if self.current_chat_id in self.chats:
                self.chats[self.current_chat_id]["is_generating"] = False
            if self.current_chat_id in self.chat_workers:
                try:
                    self.chat_workers[self.current_chat_id].token_received.disconnect()
                    self.chat_workers[self.current_chat_id].finished.disconnect()
                except:
                    pass
                del self.chat_workers[self.current_chat_id]
            
        self.update_ui_state()
        self.save_current_chat()
        
    def on_profile_changed(self):
        """Handle profile selection change"""
        if not self.current_chat_id:
            return
            
        # Get selected profile
        profile_id = self.profile_combo.currentData()
        if not profile_id:
            return
            
        # Update current chat's profile
        self.chats[self.current_chat_id]["profile"] = profile_id
        profile = self.profiles[profile_id]
        
        # Update system prompt and sidebar parameters
        self.system_prompt.setPlainText(profile["system_prompt"])
        self.temp_value.setText(f"{profile['temperature']:.2f}")
        self.temp_slider.setValue(int(profile['temperature'] * 100))
        self.token_spinbox.setValue(profile['max_tokens'])
        
        # Save the chat with new profile
        self.save_current_chat()
        
    def update_profile_ui(self):
        """Update profile UI to match current chat"""
        if not self.current_chat_id or self.current_chat_id not in self.chats:
            return
            
        profile_id = self.chats[self.current_chat_id].get("profile", "general")
        
        # Update combo box selection
        for i in range(self.profile_combo.count()):
            if self.profile_combo.itemData(i) == profile_id:
                self.profile_combo.setCurrentIndex(i)
                break
                
        # Update system prompt and sidebar parameters
        profile = self.profiles.get(profile_id, self.profiles["general"])
        self.system_prompt.setPlainText(profile["system_prompt"])
        
        # Update sidebar parameter displays
        self.temp_value.setText(f"{profile['temperature']:.2f}")
        self.temp_slider.setValue(int(profile['temperature'] * 100))
        self.token_spinbox.setValue(profile['max_tokens'])
        
    def new_chat(self):
        self.chat_counter += 1
        chat_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.chat_counter}"
        
        # Save current chat if exists
        if self.current_chat_id:
            self.save_current_chat()
        
        # Create new chat
        chat_data = {
            "id": chat_id,
            "name": "New Chat",
            "messages": [],
            "html_content": "",
            "created": datetime.now().isoformat(),
            "is_generating": False,
            "profile": "general",  # Default profile
            "draft_message": ""  # Initialize draft message field
        }
        self.chats[chat_id] = chat_data
        self.current_chat_id = chat_id
        
        # Add to list
        item = QListWidgetItem("New Chat")
        item.setData(Qt.UserRole, chat_id)
        self.chat_list.addItem(item)
        self.chat_list.setCurrentItem(item)
        
        # Clear display and update UI
        self.chat_display.clear()
        self.update_profile_ui()  # Load profile for new chat
        self.update_ui_state()
        
        return chat_id
        
    def switch_chat(self, item):
        # Save current chat and input field text
        if self.current_chat_id:
            # Save input field text for current chat (with safety check)
            if self.current_chat_id in self.chats:
                self.chats[self.current_chat_id]["draft_message"] = self.input_field.text()
            self.save_current_chat()
        
        # Switch to selected chat
        chat_id = item.data(Qt.UserRole)
        if chat_id not in self.chats:
            return
            
        self.current_chat_id = chat_id
        chat_data = self.chats[chat_id]
        
        # Restore input field text for this chat
        draft_message = chat_data.get("draft_message", "")
        self.input_field.setText(draft_message)
        
        # Restore chat display from saved HTML content
        self.chat_display.clear()
        
        # Use saved HTML content if available, otherwise rebuild from messages
        html_content = chat_data.get("html_content", "")
        if html_content:
            self.chat_display.setHtml(html_content)
        else:
            # Fallback: rebuild from messages with proper formatting
            from PySide6.QtGui import QTextBlockFormat, QTextCursor
            doc = self.chat_display.document()
            cursor = self.chat_display.textCursor()
            block_format = QTextBlockFormat()
            block_format.setLineHeight(120, 1)  # 1 = ProportionalHeight
            block_format.setBottomMargin(8)
            
            for message in chat_data.get("messages", []):
                role = message.get("role", "")
                content = message.get("content", "")
                
                cursor.movePosition(QTextCursor.End)
                cursor.setBlockFormat(block_format)
                
                if role == "user":
                    cursor.insertHtml(f"<b>You:</b> {content}<br><br>")
                elif role == "assistant":
                    cursor.insertHtml(f"<b>AI:</b> {content}<br><br>")
        
        # Update profile UI for this chat
        self.update_profile_ui()
        
        # Update UI state for this chat
        self.update_ui_state()
        
        # Refresh document list
        self.refresh_document_list()
        
    def delete_current_chat(self):
        """Delete the currently selected chat with confirmation"""
        if not self.current_chat_id or self.current_chat_id not in self.chats:
            return
            
        # Show confirmation dialog
        chat_name = self.chats[self.current_chat_id].get('name', 'this chat')
        reply = QMessageBox.question(
            self, 
            "Delete Chat", 
            f"Are you sure you want to delete '{chat_name}'?\n\nThis action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
            
        # Stop any ongoing generation for this chat
        if self.current_chat_id in self.chat_workers:
            worker = self.chat_workers[self.current_chat_id]
            if worker.isRunning():
                worker.stop()
            del self.chat_workers[self.current_chat_id]
        
        # Remove chat file
        file_path = os.path.join(self.chats_dir, f"{self.current_chat_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Remove from chat list widget
        for i in range(self.chat_list.count()):
            item = self.chat_list.item(i)
            if item.data(Qt.UserRole) == self.current_chat_id:
                self.chat_list.takeItem(i)
                break
        
        # Remove from chats dictionary
        del self.chats[self.current_chat_id]
        
        # Switch to another chat or create new one
        if self.chat_list.count() > 0:
            # Select the first available chat
            first_item = self.chat_list.item(0)
            self.chat_list.setCurrentItem(first_item)
            self.switch_chat(first_item)
        else:
            # No chats left, create a new one
            self.current_chat_id = None
            self.chat_display.clear()
            self.new_chat()
        
        self.update_ui_state()
        
    def update_ui_state(self):
        """Update UI buttons based on current chat's generation state"""
        has_chat = self.current_chat_id is not None
        has_model = self.llama_handler is not None
        
        if not has_chat or not has_model:
            self.send_button.setEnabled(False)
            self.stop_button.setEnabled(False)
        else:
            is_generating = self.chats[self.current_chat_id]["is_generating"]
            self.send_button.setEnabled(not is_generating)
            self.stop_button.setEnabled(is_generating)
        
        # Enable delete button only if there's a current chat
        self.delete_chat_button.setEnabled(has_chat)
        
    def save_current_chat(self):
        if self.current_chat_id:
            self.save_chat_by_id(self.current_chat_id)
            
    def save_chat_by_id(self, chat_id):
        if chat_id not in self.chats:
            return
            
        chat_data = self.chats[chat_id]
        
        # Update HTML content if this is the current chat
        if chat_id == self.current_chat_id:
            chat_data["html_content"] = self.chat_display.toHtml()
        
        # Save to JSON file with atomic write
        file_path = os.path.join(self.chats_dir, f"{chat_id}.json")
        temp_path = file_path + ".tmp"
        
        # Create a copy without the is_generating flag for saving
        save_data = {k: v for k, v in chat_data.items() if k != "is_generating"}
        
        try:
            # Write to temporary file first
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename (Windows-safe)
            if os.path.exists(file_path):
                os.remove(file_path)
            os.rename(temp_path, file_path)
            
        except PermissionError:
            logger.error(f"Permission denied saving chat {chat_id}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError as e:
            logger.error(f"OS error saving chat {chat_id}: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            logger.error(f"Unexpected error saving chat {chat_id}: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
    def load_existing_chats(self):
        if not os.path.exists(self.chats_dir):
            return
            
        for filename in os.listdir(self.chats_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.chats_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    
                    # Skip empty files
                    if not content:
                        logger.warning(f"Skipping empty chat file {filename}")
                        continue
                        
                    chat_data = json.loads(content)
                        
                    # Validate required fields
                    required_fields = ["id", "name", "html_content", "created"]
                    if not all(field in chat_data for field in required_fields):
                        logger.warning(f"Skipping invalid chat file {filename}: missing required fields")
                        continue
                    
                    chat_id = chat_data["id"]
                    # Initialize generation state
                    chat_data["is_generating"] = False
                    # Add default profile if missing (backward compatibility)
                    if "profile" not in chat_data:
                        chat_data["profile"] = "general"
                    # Add default draft_message if missing
                    if "draft_message" not in chat_data:
                        chat_data["draft_message"] = ""
                    self.chats[chat_id] = chat_data
                    
                    # Add to list
                    item = QListWidgetItem(chat_data["name"])
                    item.setData(Qt.UserRole, chat_id)
                    self.chat_list.addItem(item)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in chat file {filename}: {e}")
                except FileNotFoundError:
                    logger.warning(f"Chat file {filename} not found")
                except PermissionError:
                    logger.error(f"Permission denied reading chat file {filename}")
                except Exception as e:
                    logger.error(f"Error loading chat {filename}: {e}")
                    
    def update_chat_name(self, user_message):
        if self.current_chat_id and self.chats[self.current_chat_id]["name"] == "New Chat":
            # Use first 50 chars of user message as chat name
            chat_name = user_message[:50] + "..." if len(user_message) > 50 else user_message
            self.chats[self.current_chat_id]["name"] = chat_name
            
            # Update the list item
            for i in range(self.chat_list.count()):
                item = self.chat_list.item(i)
                if item.data(Qt.UserRole) == self.current_chat_id:
                    item.setText(chat_name)
                    break
                    
    def clear_chat(self):
        self.chat_display.clear()
        if self.current_chat_id:
            self.chats[self.current_chat_id]["html_content"] = ""
            self.chats[self.current_chat_id]["messages"] = []
            self.save_current_chat()
            
    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # Delete chat shortcut (Ctrl+D)
        delete_shortcut = QShortcut(QKeySequence("Ctrl+D"), self)
        delete_shortcut.activated.connect(self.delete_current_chat)
        
        # New chat shortcut (Ctrl+N)
        new_chat_shortcut = QShortcut(QKeySequence("Ctrl+N"), self)
        new_chat_shortcut.activated.connect(self.new_chat)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
            
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.drag_position:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()
            
    def show_chat_context_menu(self, position):
        item = self.chat_list.itemAt(position)
        if not item:
            return
            
        menu = QMenu(self)
        
        clear_action = QAction("Clear Chat", self)
        clear_action.triggered.connect(self.clear_chat)
        menu.addAction(clear_action)
        
        delete_action = QAction("Delete Chat", self)
        delete_action.triggered.connect(self.delete_current_chat)
        menu.addAction(delete_action)
        
        menu.exec(self.chat_list.mapToGlobal(position))
        
    def update_temperature(self, value):
        self.temperature = value / 100.0
        self.temp_value.setText(f"{self.temperature:.2f}")
        # Clear preset button selection when slider is moved manually
        self.clear_preset_selection()
    
    def set_temperature_preset(self, temp_value):
        self.temperature = temp_value
        self.temp_value.setText(f"{self.temperature:.2f}")
        # Update slider without triggering valueChanged signal
        self.temp_slider.blockSignals(True)
        self.temp_slider.setValue(int(temp_value * 100))
        self.temp_slider.blockSignals(False)
        # Update preset button styles
        self.update_preset_selection(temp_value)
    
    def update_preset_selection(self, temp_value):
        # Reset all buttons
        self.robotic_btn.setStyleSheet("font-size: 10px;")
        self.default_btn.setStyleSheet("font-size: 10px;")
        self.creative_btn.setStyleSheet("font-size: 10px;")
        
        # Highlight active preset
        if temp_value == 0.2:
            self.robotic_btn.setStyleSheet("font-size: 10px; background-color: #4CAF50; color: white;")
        elif temp_value == 0.7:
            self.default_btn.setStyleSheet("font-size: 10px; background-color: #4CAF50; color: white;")
        elif temp_value == 1.0:
            self.creative_btn.setStyleSheet("font-size: 10px; background-color: #4CAF50; color: white;")
    
    def clear_preset_selection(self):
        self.robotic_btn.setStyleSheet("font-size: 10px;")
        self.default_btn.setStyleSheet("font-size: 10px;")
        self.creative_btn.setStyleSheet("font-size: 10px;")
        
    def update_tokens(self, value):
        self.max_tokens = value
    
    def export_current_chat(self):
        """Export the current chat to a Markdown file"""
        if not self.current_chat_id or self.current_chat_id not in self.chats:
            QMessageBox.information(self, "Export Chat", "No active chat to export.")
            return
        
        # Generate default filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"localmind_chat_{timestamp}.md"
        
        # Open file save dialog
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Chat",
            default_filename,
            "Markdown Files (*.md);;All Files (*)"
        )
        
        if not filename:
            return
        
        try:
            chat_data = self.chats[self.current_chat_id]
            
            # Get current profile info
            current_profile = chat_data.get('profile', 'general')
            profile_name = self.profiles.get(current_profile, {}).get('name', 'Unknown')
            
            # Get model name
            model_name = os.path.basename(self.model_path) if self.model_path else "No model loaded"
            
            # Create markdown content
            markdown_content = f"""# LocalMind Chat Export

**Chat:** {chat_data.get('name', 'Untitled Chat')}  
**Profile:** {profile_name}  
**Model:** {model_name}  
**Exported:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  

---

"""
            
            # Add messages
            messages = chat_data.get('messages', [])
            for msg in messages:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                
                if role == 'user':
                    markdown_content += f"## User\n\n{content}\n\n"
                elif role == 'assistant':
                    markdown_content += f"## Assistant\n\n{content}\n\n"
            
            # Write to file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            QMessageBox.information(self, "Export Successful", f"Chat exported to:\n{filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export chat:\n{str(e)}")
    
    def start_model_download(self):
        """Start model download with cancel support"""
        self.download_btn.setEnabled(False)
        self.download_btn.setText("Downloading model…")
        self.cancel_btn.setVisible(True)
        self.update_download_progress(0)
        
        # Create models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        self.downloader = ModelDownloader(
            repo_id="hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF",
            filename="llama-3.2-3b-instruct-q4_k_m.gguf",
            local_dir=str(models_dir)
        )
        
        self.downloader.progress_updated.connect(self.update_download_progress)
        self.downloader.download_finished.connect(self.download_finished)
        self.downloader.download_failed.connect(self.download_failed)
        self.downloader.start()
    
    def cancel_download(self):
        """Cancel download cooperatively"""
        if self.downloader:
            self.downloader.request_cancel()
        
        self.download_btn.setText("Download cancelled")
        self.download_btn.setEnabled(True)
        self.download_btn.setStyleSheet(self.IDLE_STYLE)
        self.cancel_btn.setVisible(False)
    
    def update_download_progress(self, percent, status=None):
        """Update download progress with button fill animation"""
        p = percent / 100.0
        style = self.DOWNLOADING_STYLE.format(progress=p)
        self.download_btn.setStyleSheet(style)
    
    def download_finished(self, model_path):
        """Handle successful download completion"""
        self.download_btn.setText("Model ready")
        self.download_btn.setStyleSheet(self.DONE_STYLE)
        self.cancel_btn.setVisible(False)
        
        # Auto-enable and set model path
        self.model_path = model_path
        self.load_button.setEnabled(True)
        self.save_settings()
        
        # Update model status
        model_name = os.path.basename(model_path).replace('.gguf', '')
        self.model_status.setText(f"Selected: {model_name}")
    
    def download_failed(self, error_message):
        """Handle download failure"""
        self.download_btn.setText("Download failed")
        self.download_btn.setEnabled(True)
        self.download_btn.setStyleSheet(self.IDLE_STYLE)
        self.cancel_btn.setVisible(False)
        
        # Show error in model status
        self.model_status.setText(f"Download failed: {error_message}")

    def show_download_model_dialog(self):
        """Legacy method - now handled by sidebar button"""
        pass
    
    # Document Methods
    def import_document(self):
        """Import and learn from a new document"""
        try:
            if not self.llama_handler:
                QMessageBox.warning(self, "No Model", 
                    "Please load a model first to learn from documents.")
                return
            
            file_dialog = QFileDialog()
            file_dialog.setFileMode(QFileDialog.ExistingFile)
            file_dialog.setNameFilter("Documents (*.txt *.pdf *.docx *.md)")
            
            if file_dialog.exec():
                file_paths = file_dialog.selectedFiles()
                if file_paths:
                    file_path = file_paths[0]
                    
                    # Show progress bar
                    progress = QProgressDialog("Processing document...", None, 0, 100, self)
                    progress.setWindowTitle("LocalMind")
                    progress.setWindowModality(Qt.WindowModal)
                    progress.setCancelButton(None)
                    progress.setMinimumDuration(0)
                    progress.setValue(10)
                    progress.show()
                    QApplication.processEvents()
                    
                    try:
                        # Update progress during processing
                        progress.setLabelText("Extracting text...")
                        progress.setValue(30)
                        QApplication.processEvents()
                        
                        # Learn document with progress callback
                        doc_id = self.memory_manager.learn_document(
                            file_path,
                            progress_callback=lambda msg, val: self._update_progress(progress, msg, val)
                        )
                        
                        progress.setValue(100)
                        progress.close()
                        
                        if doc_id:
                            self.refresh_document_list()
                            
                            QMessageBox.information(self, "Success", 
                                "Document learned and compressed into memory!")
                        else:
                            QMessageBox.warning(self, "Error", 
                                "Failed to learn document. Check logs.")
                    except Exception as e:
                        progress.close()
                        logger.error(f"Error learning document: {e}")
                        QMessageBox.critical(self, "Error", f"Failed to learn document: {str(e)}")
        except Exception as e:
            logger.error(f"Error in import_document: {e}")
            QMessageBox.critical(self, "Error", f"Failed to open file dialog: {str(e)}")
    
    def show_compression_stats(self, stats, doc_name):
        """Show compression statistics after successful document import"""
        msg = QMessageBox(self)
        msg.setWindowTitle("Document Compressed Successfully!")
        msg.setIcon(QMessageBox.Information)
        
        stats_text = f"""📄 {doc_name}

📊 Original size: {stats['original_kb']:.1f} KB
📦 Compressed size: {stats['compressed_kb']:.1f} KB
🎯 Compression ratio: {stats['ratio']:.1f}x
💾 Space saved: {stats['savings_percent']:.1f}%

✅ Document learned and ready for chat!"""
        
        msg.setText(stats_text)
        msg.exec()

    def _update_progress(self, progress, message, value):
        """Update progress dialog"""
        progress.setLabelText(message)
        progress.setValue(value)
        QApplication.processEvents()
    
    def refresh_document_list(self):
        """Refresh the document list"""
        self.document_list.clear()
        
        try:
            memories = self.memory_manager.list_learned_documents()
            
            if not memories:
                item = QListWidgetItem("No documents learned yet")
                item.setData(Qt.UserRole, None)
                self.document_list.addItem(item)
                return
            
            for mem in memories:
                item_text = f"📚 {mem['name']}\n   {mem['summary']}"
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, mem['id'])
                self.document_list.addItem(item)
                
        except Exception as e:
            logger.error(f"Error refreshing document list: {e}")
            item = QListWidgetItem("Error loading documents")
            item.setData(Qt.UserRole, None)
            self.document_list.addItem(item)
    
    def on_document_selected(self, item):
        """Handle document selection"""
        doc_id = item.data(Qt.UserRole)
        if doc_id:
            self.set_active_document(doc_id)
    
    def set_active_document(self, doc_id: str):
        """No longer needed - all learned documents are always available"""
        pass
    
    def update_active_document_display(self):
        """No longer needed - memory is always active"""
        pass
    
    def show_document_selection(self):
        """No longer needed - all documents in memory are used"""
        pass
    
    def clear_active_document(self):
        """No longer needed - memory persists"""
        pass
    
    def show_document_context_menu(self, position):
        """Show context menu for documents"""
        item = self.document_list.itemAt(position)
        if not item:
            return
        
        doc_id = item.data(Qt.UserRole)
        if not doc_id:
            return
        
        menu = QMenu(self)
        delete_action = menu.addAction("Forget")
        
        action = menu.exec(self.document_list.mapToGlobal(position))
        
        if action == delete_action:
            self.delete_document(doc_id)
    
    def delete_document(self, doc_id: str):
        """Forget a document from memory"""
        reply = QMessageBox.question(
            self, "Forget Document", 
            "Remove this document from memory?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                self.memory_manager.forget_document(doc_id)
                self.refresh_document_list()
                QMessageBox.information(self, "Success", "Document removed from memory.")
            except Exception as e:
                logger.error(f"Error forgetting document: {e}")
                QMessageBox.critical(self, "Error", f"Failed to forget document: {str(e)}")
    
    
    def closeEvent(self, event):
        """Clean up resources when closing the application"""
        # Check for active download first
        if self.download_worker and self.download_worker.isRunning():
            QMessageBox.warning(
                self,
                "Download in progress",
                "A model is still downloading.\nPlease cancel it first."
            )
            event.ignore()
            return
        
        try:
            # Save current chat first
            if self.current_chat_id:
                self.save_current_chat()
            
            # Stop all active chat workers with proper cleanup
            workers_to_stop = list(self.chat_workers.items())
            for chat_id, worker in workers_to_stop:
                try:
                    if worker and worker.isRunning():
                        worker.stop()
                        if not worker.wait(2000):  # Wait max 2 seconds
                            logger.warning(f"Force terminating worker {chat_id}")
                            worker.terminate()
                            worker.wait(1000)
                        worker.deleteLater()
                except Exception as e:
                    logger.error(f"Error stopping worker {chat_id}: {e}")
            
            # Clear workers dict
            self.chat_workers.clear()
            
            # Stop model loader if running
            try:
                if hasattr(self, 'model_loader') and self.model_loader:
                    if self.model_loader.isRunning():
                        if not self.model_loader.wait(2000):
                            self.model_loader.terminate()
                            self.model_loader.wait(1000)
                        self.model_loader.deleteLater()
                    self.model_loader = None
            except Exception as e:
                logger.error(f"Error stopping model loader: {e}")
            
            # Cleanup model
            try:
                if hasattr(self, 'llama_handler') and self.llama_handler:
                    self.llama_handler.cleanup()
                    self.llama_handler = None
            except Exception as e:
                logger.error(f"Error cleaning up model: {e}")
            
            # Clear data structures
            try:
                self.chats.clear()
            except Exception as e:
                logger.error(f"Error clearing data: {e}")
                
        except Exception as e:
            logger.error(f"Error in closeEvent: {e}")
        finally:
            # Always accept the close event
            event.accept()
