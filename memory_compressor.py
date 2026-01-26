"""
Memory Compressor - CLaRa-inspired document learning system
Documents are digested once into compressed knowledge, not retrieved per query.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class CompressedMemory:
    """Compressed knowledge from a document"""
    doc_id: str
    doc_name: str
    summary: str  # High-level overview
    key_concepts: List[str]  # Main ideas/terms
    facts: List[str]  # Important facts/statements
    glossary: Dict[str, str]  # Term definitions
    structure: str  # Document organization
    raw_text: str  # For exact quotes only


class MemoryCompressor:
    """Compresses documents into bounded long-term memory"""
    
    def __init__(self, memory_dir: str = "memory", max_memory_kb: int = 2000):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        self.memory_file = self.memory_dir / "compressed_memory.json"
        
        # Windows-compatible memory limits
        import platform
        if platform.system() == "Windows":
            self.max_memory_kb = min(max_memory_kb, 1000)  # Lower limit for Windows
        else:
            self.max_memory_kb = max_memory_kb  # Full limit for Linux/Mac
            
        self.memories: Dict[str, CompressedMemory] = {}
        self._load_memories()
    
    def _load_memories(self):
        """Load compressed memories from disk"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for doc_id, mem_dict in data.items():
                        self.memories[doc_id] = CompressedMemory(**mem_dict)
                logger.info(f"Loaded {len(self.memories)} compressed memories")
            except Exception as e:
                logger.error(f"Failed to load memories: {e}")
    
    def _save_memories(self):
        """Save compressed memories to disk"""
        try:
            data = {doc_id: asdict(mem) for doc_id, mem in self.memories.items()}
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save memories: {e}")
    
    def _check_memory_bounds(self):
        """Ensure memory stays within bounds"""
        current_size = self.memory_file.stat().st_size / 1024 if self.memory_file.exists() else 0
        
        logger.info(f"Memory check: {current_size:.1f}KB / {self.max_memory_kb}KB limit")
        
        if current_size > self.max_memory_kb:
            # Remove oldest memories until under 90% of limit (more conservative)
            sorted_ids = sorted(self.memories.keys())
            target_size = self.max_memory_kb * 0.9
            
            logger.warning(f"Memory limit exceeded, cleaning up to {target_size:.1f}KB")
            
            while current_size > target_size and sorted_ids:
                oldest = sorted_ids.pop(0)
                del self.memories[oldest]
                logger.info(f"Removed memory {oldest} to stay within bounds")
                # Recalculate size after removal
                self._save_memories()
                current_size = self.memory_file.stat().st_size / 1024 if self.memory_file.exists() else 0
    
    def compress_document(self, doc_id: str, doc_name: str, text: str, llm_handler, progress_callback=None) -> CompressedMemory:
        """CLaRa-inspired compression"""
        logger.info(f"Compressing document: {doc_name}")
        
        if progress_callback:
            progress_callback("Compressing with LLM...", 60)
        
        # Truncate text to fit in context
        text_sample = text[:6000] if len(text) > 6000 else text
        
        prompt = f"""Extract key information from this document:

{text_sample}

Respond with:
1. A brief summary (2-3 sentences)
2. Main concepts or topics (list 5-10)
3. Important facts or key points (list 5-10)"""

        compressed = ""
        try:
            for token in llm_handler.generate(
                prompt,
                system_prompt="You extract key information from documents concisely.",
                temperature=0.2,
                max_tokens=600
            ):
                compressed += token
            
            if progress_callback:
                progress_callback("Parsing compression...", 85)
            
            # Parse with flexible extraction
            memory = self._flexible_parse(doc_id, doc_name, compressed, text)
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            memory = self._fallback_compression(doc_id, doc_name, text)
        
        # Store and save
        self.memories[doc_id] = memory
        self._save_memories()
        self._check_memory_bounds()
        
        logger.info(f"Compressed {doc_name} into memory")
        return memory
    
    def _flexible_parse(self, doc_id: str, doc_name: str, compressed: str, raw_text: str) -> CompressedMemory:
        """Flexible parsing that extracts whatever the LLM provides"""
        lines = [l.strip() for l in compressed.split('\n') if l.strip()]
        
        summary = ""
        key_concepts = []
        facts = []
        
        # Extract summary (usually first few lines)
        summary_lines = []
        for i, line in enumerate(lines[:5]):
            if not line.startswith(('-', '•', '*', '1.', '2.')) and len(line) > 20:
                summary_lines.append(line)
        summary = ' '.join(summary_lines[:3]) if summary_lines else lines[0] if lines else "Document imported"
        
        # Extract concepts and facts from bullet points
        for line in lines:
            # Skip headers
            if any(x in line.lower() for x in ['summary:', 'concept:', 'fact:', 'point:']):
                continue
            
            # Extract from bullet points
            if line.startswith(('-', '•', '*')):
                item = line.lstrip('-•* ').strip()
                if len(item) > 10:
                    if len(item) < 50:  # Short = concept
                        key_concepts.append(item)
                    else:  # Long = fact
                        facts.append(item)
            # Extract from numbered lists
            elif line[0].isdigit() and '.' in line[:3]:
                item = line.split('.', 1)[1].strip()
                if len(item) > 10:
                    if len(item) < 50:
                        key_concepts.append(item)
                    else:
                        facts.append(item)
        
        # Fallback: extract from raw text if nothing found
        if not key_concepts and not facts:
            words = raw_text.split()[:100]
            key_concepts = [w.strip('.,!?') for w in words if len(w) > 6][:10]
            facts = [raw_text[:200]]
        
        return CompressedMemory(
            doc_id=doc_id,
            doc_name=doc_name,
            summary=summary[:500],
            key_concepts=key_concepts[:15],
            facts=facts[:20],
            glossary={},
            structure="Compressed",
            raw_text=raw_text[:10000]
        )
    
    def _fallback_compression(self, doc_id: str, doc_name: str, text: str) -> CompressedMemory:
        """Fallback compression without LLM"""
        lines = text.split('\n')
        first_para = ' '.join(lines[:5])
        
        return CompressedMemory(
            doc_id=doc_id,
            doc_name=doc_name,
            summary=first_para[:300],
            key_concepts=[],
            facts=[],
            glossary={},
            structure="Document structure not analyzed",
            raw_text=text[:10000]
        )
    
    def get_compression_stats(self, doc_id: str) -> Optional[Dict]:
        """Get compression statistics for a document"""
        if doc_id not in self.memories:
            return None
        
        memory = self.memories[doc_id]
        original_size = len(memory.raw_text.encode('utf-8'))
        compressed_data = asdict(memory)
        compressed_size = len(json.dumps(compressed_data).encode('utf-8'))
        
        if compressed_size == 0:
            return None
            
        ratio = original_size / compressed_size
        savings_percent = ((original_size - compressed_size) / original_size) * 100
        
        return {
            'original_kb': original_size / 1024,
            'compressed_kb': compressed_size / 1024,
            'ratio': ratio,
            'savings_percent': savings_percent
        }

    def get_memory_context(self, doc_ids: Optional[List[str]] = None) -> str:
        """
        Build context from compressed memories.
        This is injected into the system prompt, not retrieved per query.
        """
        if not self.memories:
            return ""
        
        # Use all memories or specific ones
        target_memories = []
        if doc_ids:
            target_memories = [self.memories[did] for did in doc_ids if did in self.memories]
        else:
            target_memories = list(self.memories.values())
        
        if not target_memories:
            return ""
        
        # Build compressed context
        context_parts = ["=== LEARNED KNOWLEDGE ===\n"]
        
        for mem in target_memories:
            context_parts.append(f"\nFrom: {mem.doc_name}")
            context_parts.append(f"Summary: {mem.summary}")
            
            if mem.key_concepts:
                context_parts.append(f"Key Concepts: {', '.join(mem.key_concepts[:5])}")
            
            if mem.facts:
                context_parts.append("Important Facts:")
                for fact in mem.facts[:5]:
                    context_parts.append(f"  • {fact}")
            
            if mem.glossary:
                context_parts.append("Definitions:")
                for term, defn in list(mem.glossary.items())[:3]:
                    context_parts.append(f"  • {term}: {defn}")
        
        context_parts.append("\n=== END LEARNED KNOWLEDGE ===")
        return '\n'.join(context_parts)
    
    def find_citation(self, doc_id: str, query: str) -> Optional[str]:
        """Find exact quote from raw text for citations only"""
        if doc_id not in self.memories:
            return None
        
        raw_text = self.memories[doc_id].raw_text
        query_lower = query.lower()
        
        # Simple substring search for exact quotes
        if query_lower in raw_text.lower():
            idx = raw_text.lower().index(query_lower)
            start = max(0, idx - 100)
            end = min(len(raw_text), idx + len(query) + 100)
            return raw_text[start:end]
        
        return None
    
    def remove_memory(self, doc_id: str):
        """Remove a document's memory"""
        if doc_id in self.memories:
            del self.memories[doc_id]
            self._save_memories()
            logger.info(f"Removed memory: {doc_id}")
    
    def list_memories(self) -> List[Dict]:
        """List all compressed memories"""
        return [
            {
                "id": doc_id,
                "name": mem.doc_name,
                "summary": mem.summary[:100] + "..." if len(mem.summary) > 100 else mem.summary,
                "concepts_count": len(mem.key_concepts),
                "facts_count": len(mem.facts)
            }
            for doc_id, mem in self.memories.items()
        ]
