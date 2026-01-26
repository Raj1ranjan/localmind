# LocalMind - Memory-Based AI Chat Application 🧠



LocalMind is a **revolutionary desktop AI chat application** that challenges the traditional RAG paradigm. Built with PySide6 and llama-cpp-python, it features a **CLaRa-inspired memory-based learning system** where documents are compressed once into bounded long-term memory, enabling direct generation from learned knowledge without retrieval overhead.

> **"Documents are teachers, not databases"** - LocalMind's core philosophy

## 🚀 Quick Start

### Minimal Installation (2 dependencies only!)

```bash
pip install PySide6 llama-cpp-python
python main.py
```

### Full Installation

```bash
git clone https://github.com/user/localmind
cd localmind
pip install -r requirements.txt
python main.py
```

## 🧠 Memory-Based Architecture

**Documents are teachers, not databases.**

Instead of traditional RAG (chunk → embed → retrieve), LocalMind uses **compression-based learning**:

1. **Import document** → LLM compresses into structured knowledge (summary, concepts, facts, glossary)
2. **Store in memory** → Bounded 500KB storage with auto-cleanup
3. **Chat** → Model answers from learned knowledge (no retrieval overhead)

### Key Benefits

✅ **No retrieval overhead** - Direct generation from memory  
✅ **Simpler architecture** - No embeddings or vector search  
✅ **Bounded memory** - Predictable resource usage (500KB limit)  
✅ **Better coherence** - Holistic understanding vs fragmented chunks  
✅ **Fewer dependencies** - Just LLM, no sentence-transformers/faiss  
✅ **Always available** - All learned documents accessible without selection

## 📋 Supported File Types

- `.txt` - Plain text files
- `.pdf` - PDF documents (requires PyPDF2)
- `.docx` - Word documents (requires python-docx)
- `.md` - Markdown files

## 🎯 Features

### Memory System
- **One-time compression** - Documents learned once, not retrieved per query
- **Structured knowledge** - Summary, key concepts, facts, glossary extracted
- **Bounded storage** - 500KB limit with automatic cleanup
- **No document selection** - All memories always available

### Chat Profiles
- **General** - Balanced conversation (temp 0.70)
- **Document** - Precise, factual answers from learned knowledge (temp 0.30)
- **Student** - Clear explanations with examples (temp 0.60)
- **Code** - Programming assistance (temp 0.30)
- **Writer** - Creative writing (temp 0.80)

### Core Features
- **Streaming responses** - Real-time token generation
- **Multiple chats** - Manage separate conversations
- **Chat persistence** - Save/load conversations
- **Model flexibility** - Works with any GGUF model
- **Dark theme** - Professional UI design

## 🔧 Usage

### 1. Load Model
- Click "Browse Model"
- Select your GGUF model file
- Click "Load Model"
- Wait for "Model loaded successfully!"

### 2. Import Documents
- Click "Import Document"
- Select PDF/DOCX/TXT/MD file
- Wait for compression (30-60 seconds)
- See "Document learned and compressed into memory!"

### 3. Chat
- Type your question
- Model answers from learned knowledge
- No document selection needed
- All memories always available

### 4. Manage Memory
- View learned documents in sidebar
- Right-click → "Forget" to remove from memory
- Memory automatically cleaned when 500KB limit reached

## 🏗️ Architecture

### Core Components
- **Main Window** - PySide6-based UI
- **LLM Handler** - llama-cpp-python integration with memory context injection
- **Memory Compressor** - Compresses documents using LLM
- **Memory Manager** - Integration layer for document learning

### Memory Flow

```
Document Import:
  PDF → Extract text → LLM compresses → Store in memory/

Chat:
  Query → Load all memories → Inject into prompt → Generate
```

No retrieval. No search. No embeddings. Just learned knowledge.

## 📝 Memory Structure

```json
{
  "doc_id": {
    "doc_name": "document.pdf",
    "summary": "2-3 sentence overview",
    "key_concepts": ["concept1", "concept2"],
    "facts": ["fact1", "fact2"],
    "glossary": {"term": "definition"},
    "structure": "Document organization",
    "raw_text": "Limited text for citations"
  }
}
```

## 🔍 Troubleshooting

### "Please load a model first"
→ Load model before importing documents (compression requires LLM)

### "Failed to learn document"
→ Check `logs/localmind.log` for details

### No response from model
→ Verify model loaded and context window not exceeded

### Compression failed
→ Check if document is readable and not corrupted
→ Verify model has sufficient context window

## 📚 Documentation

- `.kiro/MEMORY_ARCHITECTURE.md` - System design and philosophy
- `.kiro/PROFILE_GUIDE.md` - When to use each profile
- `.kiro/GLOBAL_RULES.md` - Development guidelines
- `.kiro/COMPLETE.txt` - Full implementation summary

## 🎓 Philosophy

**CLaRa-Inspired Learning**

LocalMind follows the CLaRa framework philosophy:
- **Compression over retrieval** - Learn through compression, not search
- **Documents as teachers** - Not databases to query
- **Bounded memory** - Predictable, manageable knowledge storage
- **Holistic understanding** - Compressed knowledge, not fragmented chunks

## 🚫 What LocalMind Does NOT Use

❌ Chunk embeddings  
❌ Vector databases  
❌ Top-k retrieval  
❌ Per-query search  
❌ Document activation/selection  

## ✅ What LocalMind DOES Use

✅ One-time LLM compression  
✅ Structured knowledge extraction  
✅ Bounded memory storage  
✅ Direct generation from memory  
✅ All documents always available  

## 🔄 Migration from RAG

If you're familiar with traditional RAG systems:

| RAG | Memory (LocalMind) ClaRa |
|-----|-------------------|
| Chunk documents | Compress documents |
| Generate embeddings | Extract structured knowledge |
| Store in vector DB | Store in bounded JSON |
| Search per query | Inject memory once |
| Retrieve top-k chunks | Use all learned knowledge |
| Fragmented context | Holistic understanding |

## 🎯 Use Cases

- **Research** - Learn from papers, answer questions from compressed knowledge
- **Study** - Import textbooks, get explanations from learned material
- **Documentation** - Learn technical docs, answer from structured knowledge
- **Books** - Compress books, discuss concepts and ideas
- **Notes** - Import notes, query from learned information

## 🛡️ Stability Features

- **Graceful degradation** - Works with minimal dependencies
- **Error handling** - Never crashes, comprehensive logging
- **Thread safety** - Background operations don't block UI
- **Memory management** - Automatic cleanup and bounds enforcement
- **Resource cleanup** - Proper thread and model disposal

## 📊 Performance

- **Import time** - 30-60 seconds per document (one-time)
- **Query latency** - 0ms retrieval overhead (direct generation)
- **Memory usage** - Bounded to 500KB (configurable)
- **Storage** - ~50KB per document (vs ~10MB for RAG)

This ensures LocalMind is efficient, predictable, and scalable!


Demo Video-
Watch on YouTube: https://www.youtube.com/watch?v=dUMvNtizC9k
