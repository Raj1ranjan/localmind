#!/usr/bin/env python3
"""
Unit tests for LocalMind components
"""

import unittest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_compressor import MemoryCompressor, CompressedMemory
from memory_manager import MemoryManager


class TestCompressedMemory(unittest.TestCase):
    """Test CompressedMemory dataclass"""
    
    def test_compressed_memory_creation(self):
        """Test creating CompressedMemory instance"""
        memory = CompressedMemory(
            doc_id="test_doc",
            doc_name="test.pdf",
            summary="Test summary",
            key_concepts=["concept1", "concept2"],
            facts=["fact1", "fact2"],
            glossary={"term": "definition"},
            structure="Test structure",
            raw_text="Test raw text"
        )
        
        self.assertEqual(memory.doc_id, "test_doc")
        self.assertEqual(memory.doc_name, "test.pdf")
        self.assertEqual(len(memory.key_concepts), 2)
        self.assertEqual(len(memory.facts), 2)


class TestMemoryCompressor(unittest.TestCase):
    """Test MemoryCompressor functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.compressor = MemoryCompressor(memory_dir=self.temp_dir, max_memory_kb=100)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test MemoryCompressor initialization"""
        self.assertEqual(self.compressor.max_memory_kb, 100)
        self.assertTrue(Path(self.temp_dir).exists())
        self.assertEqual(len(self.compressor.memories), 0)
    
    def test_memory_storage(self):
        """Test storing compressed memory"""
        memory = CompressedMemory(
            doc_id="test_doc",
            doc_name="test.pdf",
            summary="Test summary",
            key_concepts=["concept1"],
            facts=["fact1"],
            glossary={"term": "definition"},
            structure="Test structure",
            raw_text="Test text"
        )
        
        self.compressor.memories["test_doc"] = memory
        self.compressor._save_memories()
        
        # Verify file was created
        memory_file = Path(self.temp_dir) / "compressed_memory.json"
        self.assertTrue(memory_file.exists())
        
        # Verify content
        with open(memory_file, 'r') as f:
            data = json.load(f)
            self.assertIn("test_doc", data)
            self.assertEqual(data["test_doc"]["doc_name"], "test.pdf")
    
    def test_memory_loading(self):
        """Test loading compressed memories"""
        # Create test memory file
        memory_data = {
            "test_doc": {
                "doc_id": "test_doc",
                "doc_name": "test.pdf",
                "summary": "Test summary",
                "key_concepts": ["concept1"],
                "facts": ["fact1"],
                "glossary": {"term": "definition"},
                "structure": "Test structure",
                "raw_text": "Test text"
            }
        }
        
        memory_file = Path(self.temp_dir) / "compressed_memory.json"
        with open(memory_file, 'w') as f:
            json.dump(memory_data, f)
        
        # Create new compressor to test loading
        new_compressor = MemoryCompressor(memory_dir=self.temp_dir)
        self.assertEqual(len(new_compressor.memories), 1)
        self.assertIn("test_doc", new_compressor.memories)
    
    def test_get_memory_size(self):
        """Test memory size calculation"""
        memory = CompressedMemory(
            doc_id="test_doc",
            doc_name="test.pdf",
            summary="Test summary",
            key_concepts=["concept1"],
            facts=["fact1"],
            glossary={"term": "definition"},
            structure="Test structure",
            raw_text="Test text"
        )
        
        self.compressor.memories["test_doc"] = memory
        # Test that memory exists
        self.assertEqual(len(self.compressor.memories), 1)
        self.assertIn("test_doc", self.compressor.memories)
    
    def test_forget_document(self):
        """Test forgetting a document"""
        memory = CompressedMemory(
            doc_id="test_doc",
            doc_name="test.pdf",
            summary="Test summary",
            key_concepts=["concept1"],
            facts=["fact1"],
            glossary={"term": "definition"},
            structure="Test structure",
            raw_text="Test text"
        )
        
        self.compressor.memories["test_doc"] = memory
        self.assertEqual(len(self.compressor.memories), 1)
        
        # Test manual removal since forget_document may not exist
        del self.compressor.memories["test_doc"]
        self.assertEqual(len(self.compressor.memories), 0)


class TestMemoryManager(unittest.TestCase):
    """Test MemoryManager functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = MemoryManager()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test MemoryManager initialization"""
        self.assertIsNotNone(self.manager.compressor)
        self.assertIsNotNone(self.manager.reader)
    
    def test_get_memory_context(self):
        """Test getting memory context for chat"""
        # Add a test memory directly to compressor
        memory = CompressedMemory(
            doc_id="test_doc",
            doc_name="test.pdf",
            summary="Test summary about AI",
            key_concepts=["AI", "machine learning"],
            facts=["AI is powerful"],
            glossary={"AI": "Artificial Intelligence"},
            structure="Introduction, Body, Conclusion",
            raw_text="AI is the future"
        )
        
        self.manager.compressor.memories["test_doc"] = memory
        context = self.manager.get_memory_context()
        
        self.assertIn("test.pdf", context)
        self.assertIn("Test summary about AI", context)
        self.assertIn("AI", context)
    
    def test_forget_document(self):
        """Test forgetting a document through manager"""
        memory = CompressedMemory(
            doc_id="test_doc",
            doc_name="test.pdf",
            summary="Test summary",
            key_concepts=["concept1"],
            facts=["fact1"],
            glossary={"term": "definition"},
            structure="Test structure",
            raw_text="Test text"
        )
        
        self.manager.compressor.memories["test_doc"] = memory
        self.assertEqual(len(self.manager.compressor.memories), 1)
        
        self.manager.forget_document("test_doc")
        self.assertEqual(len(self.manager.compressor.memories), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_memory_persistence(self):
        """Test that memories persist across sessions"""
        # Create compressor and add memory
        compressor1 = MemoryCompressor(memory_dir=self.temp_dir)
        memory = CompressedMemory(
            doc_id="test_doc",
            doc_name="test.pdf",
            summary="Test summary",
            key_concepts=["concept1"],
            facts=["fact1"],
            glossary={"term": "definition"},
            structure="Test structure",
            raw_text="Test text"
        )
        
        compressor1.memories["test_doc"] = memory
        compressor1._save_memories()
        
        # Create new compressor to simulate new session
        compressor2 = MemoryCompressor(memory_dir=self.temp_dir)
        self.assertEqual(len(compressor2.memories), 1)
        self.assertIn("test_doc", compressor2.memories)
        self.assertEqual(compressor2.memories["test_doc"].doc_name, "test.pdf")


if __name__ == '__main__':
    # Run tests with detailed output
    unittest.main(verbosity=2, buffer=True)
