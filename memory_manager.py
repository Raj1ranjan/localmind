"""
Integration layer for memory compression system
Replaces document_manager.py with compression-based approach
"""

import logging
import hashlib
from pathlib import Path
from typing import Optional, List, Dict
from memory_compressor import MemoryCompressor, CompressedMemory

logger = logging.getLogger(__name__)

# Document processing imports
HAS_PDF, HAS_DOCX = False, False

try:
    import PyPDF2
    HAS_PDF = True
except ImportError:
    pass

try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    pass


class DocumentReader:
    """Reads documents into text for compression"""
    
    @staticmethod
    def read_document(file_path: str) -> str:
        """Extract text from document"""
        path = Path(file_path)
        ext = path.suffix.lower()
        
        if ext == '.txt':
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        elif ext == '.md':
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        elif ext == '.pdf' and HAS_PDF:
            text = ""
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        elif ext == '.docx' and HAS_DOCX:
            doc = DocxDocument(path)
            return '\n'.join([p.text for p in doc.paragraphs])
        else:
            raise ValueError(f"Unsupported format: {ext}")


class MemoryManager:
    """
    Manages document learning through compression.
    Replaces traditional RAG document manager.
    """
    
    def __init__(self, llm_handler=None):
        self.compressor = MemoryCompressor()
        self.llm_handler = llm_handler
        self.reader = DocumentReader()
    
    def set_llm_handler(self, llm_handler):
        """Set LLM handler for compression"""
        self.llm_handler = llm_handler
    
    def learn_document(self, file_path: str, progress_callback=None) -> Optional[str]:
        """
        Learn from a document by compressing it into memory.
        Returns doc_id on success.
        """
        if not self.llm_handler:
            logger.error("No LLM handler available for compression")
            return None
        
        try:
            # Read document
            if progress_callback:
                progress_callback("Reading document...", 40)
            text = self.reader.read_document(file_path)
            
            # Generate doc_id
            doc_name = Path(file_path).name
            doc_id = hashlib.md5(doc_name.encode()).hexdigest()[:12]
            
            # Compress into memory
            if progress_callback:
                progress_callback("Compressing with LLM...", 60)
            memory = self.compressor.compress_document(
                doc_id, doc_name, text, self.llm_handler, progress_callback
            )
            
            if progress_callback:
                progress_callback("Finalizing...", 90)
            
            logger.info(f"Learned document: {doc_name}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to learn document: {e}")
            return None
    
    def get_memory_context(self) -> str:
        """
        Get all learned knowledge as context.
        This is injected into system prompt, not retrieved per query.
        """
        return self.compressor.get_memory_context()
    
    def forget_document(self, doc_id: str):
        """Remove a document from memory"""
        self.compressor.remove_memory(doc_id)
    
    def list_learned_documents(self) -> List[Dict]:
        """List all documents in memory"""
        return self.compressor.list_memories()
    
    def find_citation(self, doc_id: str, query: str) -> Optional[str]:
        """Find exact quote for citation (only use case for raw text)"""
        return self.compressor.find_citation(doc_id, query)
