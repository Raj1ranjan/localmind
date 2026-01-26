import logging
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker
from llama_cpp import Llama
import os

# Configure logging
logger = logging.getLogger("localmind.llm")
logger.setLevel(logging.INFO)

class LlamaHandler:
    def __init__(self, model_path, n_ctx=2048, n_threads=None):
        self.model_path = model_path
        # Reduce context window for Windows compatibility
        self.n_ctx = min(n_ctx, 4096)  # Cap at 4096 to avoid memory issues
        self.n_threads = n_threads or max(1, os.cpu_count() // 2)
        self.llm = None
        self.generation_lock = QMutex()
        self.is_generating = False
        
    def load_model(self):
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found: {self.model_path}")
            return False
            
        try:
            if self.llm:
                del self.llm
                self.llm = None
                
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False
            )
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            if hasattr(self, 'llm') and self.llm:
                del self.llm
                self.llm = None
            return False
    
    def interrupt(self):
        """Safely interrupt generation"""
        with QMutexLocker(self.generation_lock):
            if self.llm and self.is_generating:
                try:
                    self.llm.reset()
                    logger.info("Model interrupted safely")
                except Exception as e:
                    logger.warning(f"Failed to reset model: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.llm:
                try:
                    del self.llm
                except Exception as e:
                    logger.error(f"Error deleting model: {e}")
                finally:
                    self.llm = None
                    
                import gc
                gc.collect()
                    
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")
        finally:
            self.llm = None
    
    def build_prompt(self, user_prompt, system_prompt=None, memory_context=None):
        """Format prompt for instruction-tuned models with optional memory"""
        if system_prompt:
            # Inject memory context into system prompt
            if memory_context:
                system_prompt = f"{system_prompt}\n\n{memory_context}"
            
            return f"""### System:
{system_prompt}

### User:
{user_prompt}

### Assistant:
"""
        return user_prompt
            
    def generate(self, prompt, system_prompt=None, memory_context=None, temperature=0.7, max_tokens=512):
        with QMutexLocker(self.generation_lock):
            self.is_generating = True
        
        try:
            if not self.llm:
                logger.error("Model not loaded")
                return
            
            # Validate context window
            estimated_tokens = len(prompt.split()) * 1.3 + max_tokens
            if memory_context:
                estimated_tokens += len(memory_context.split()) * 1.3
            
            if estimated_tokens > self.n_ctx:
                logger.warning(f"Context too large ({int(estimated_tokens)} tokens), reducing max_tokens")
                max_tokens = max(100, int(self.n_ctx - len(prompt.split()) * 1.3))
            
            formatted_prompt = self.build_prompt(prompt, system_prompt, memory_context)
                
            for token in self.llm(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                stop=["### User:", "### Human:", "### Assistant:", "</s>", "\nUser:", "\nHuman:", "\n\nUser:", "\n\nHuman:", "User:", "Human:", "\n###", "Would you like", "Yes or No"]
            ):
                try:
                    yield token['choices'][0]['text']
                except (KeyError, IndexError, TypeError) as e:
                    logger.error(f"Error extracting token: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error in token generation: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Error in generate: {e}")
            return
        finally:
            with QMutexLocker(self.generation_lock):
                self.is_generating = False

class ModelLoader(QThread):
    model_loaded = Signal(bool, str)
    
    def __init__(self, model_path, n_ctx=2048, n_threads=None):
        super().__init__()
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.llama_handler = None
        
    def run(self):
        try:
            self.llama_handler = LlamaHandler(self.model_path, self.n_ctx, self.n_threads)
            success = self.llama_handler.load_model()
            message = "Model loaded successfully!" if success else "Failed to load model"
            self.model_loaded.emit(success, message)
        except Exception as e:
            self.model_loaded.emit(False, f"Error: {str(e)}")

class LlamaWorker(QThread):
    token_received = Signal(str)
    finished = Signal()
    
    def __init__(self, llama_handler, prompt, system_prompt=None, memory_context=None, temperature=0.7, max_tokens=512):
        super().__init__()
        self.llama_handler = llama_handler
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.memory_context = memory_context
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.should_stop = False
        self._running = False
        
    def stop(self):
        """Safely stop generation"""
        self.should_stop = True
        self.requestInterruption()
        
        # Interrupt model generation immediately
        if self.llama_handler and self.llama_handler.llm:
            try:
                self.llama_handler.interrupt()
            except Exception as e:
                logger.warning(f"Error interrupting model: {e}")
        
        # Wait for thread to finish
        if self._running:
            self.wait(1000)  # Wait up to 1 second only
        
    def run(self):
        self._running = True
        try:
            for token in self.llama_handler.generate(
                self.prompt, 
                self.system_prompt,
                self.memory_context,
                self.temperature, 
                self.max_tokens
            ):
                if self.should_stop or self.isInterruptionRequested():
                    break
                self.token_received.emit(token)
        except Exception as e:
            logger.error(f"Error during generation: {e}")
        finally:
            self._running = False
            self.finished.emit()
