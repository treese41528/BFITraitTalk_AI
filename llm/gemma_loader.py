# llm/gemma_loader.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base path for locally saved models
LOCAL_MODELS_PATH = "data/hf_models"

# Model configuration options
MODEL_OPTIONS = {
    "4b": "gemma-3-4b-it",
    "12b": "gemma-3-12b-it",
    "27b": "gemma-3-27b-it"
}

# Quantization configuration options
QUANTIZATION_OPTIONS = {
    "none": None,
    "8bit": "8bit",
    "4bit": "4bit"
}

# Default values
DEFAULT_MODEL_SIZE = "4b"  # Use 4B for wider compatibility
DEFAULT_QUANTIZATION = "4bit"  # 4-bit quantization for memory efficiency
DEFAULT_DEVICE = "auto"  # Auto-detect device (CUDA/CPU)

class GemmaModelManager:
    """
    Class to manage loading and configuration of Gemma 3 models.
    """
    
    def __init__(self, model_size=None, quantization=None, device=None, use_flash_attention=True):
        """
        Initialize the model manager with configuration settings.
        
        Args:
            model_size (str): Size of model ('4b', '12b', '27b')
            quantization (str): Quantization method ('none', '8bit', '4bit')
            device (str): Device to use ('auto', 'cuda', 'cpu')
            use_flash_attention (bool): Whether to use Flash Attention if available
        """
        self.model_size = model_size or DEFAULT_MODEL_SIZE
        self.quantization = quantization or DEFAULT_QUANTIZATION
        self.device = device or DEFAULT_DEVICE
        self.use_flash_attention = use_flash_attention
        
        self.model_name = MODEL_OPTIONS.get(self.model_size)
        if not self.model_name:
            logger.error(f"Invalid model size: {self.model_size}. Using default: {DEFAULT_MODEL_SIZE}")
            self.model_size = DEFAULT_MODEL_SIZE
            self.model_name = MODEL_OPTIONS[DEFAULT_MODEL_SIZE]
            
        self.quant_option = QUANTIZATION_OPTIONS.get(self.quantization)
        if self.quantization not in QUANTIZATION_OPTIONS:
            logger.error(f"Invalid quantization: {self.quantization}. Using default: {DEFAULT_QUANTIZATION}")
            self.quantization = DEFAULT_QUANTIZATION
            self.quant_option = QUANTIZATION_OPTIONS[DEFAULT_QUANTIZATION]
        
        # Local model path
        self.model_path = os.path.join(LOCAL_MODELS_PATH, self.model_name)
        
        # Check if the model exists locally
        if not os.path.exists(self.model_path):
            logger.warning(f"Model path not found: {self.model_path}")
            
        self.model = None
        self.tokenizer = None
        
    def load_model_and_tokenizer(self):
        """
        Load the Gemma 3 model and tokenizer with specified configuration.
        
        Returns:
            tuple: (model, tokenizer) or (None, None) if loading fails
        """
        logger.info(f"Loading Gemma 3 model from local path: {self.model_path}")
        logger.info(f"Quantization: {self.quantization}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Using Flash Attention: {self.use_flash_attention}")
        
        try:
            # Find the actual model files in the HF cache structure
            actual_model_path = self.model_path
            
            # Check for HF cache structure
            cache_dir_name = f"models--google--{self.model_name}"
            cache_dir_path = os.path.join(self.model_path, cache_dir_name)
            
            # Check if we're in the nested structure
            if os.path.exists(cache_dir_path):
                # Check for snapshots
                snapshot_dir = os.path.join(cache_dir_path, "snapshots")
                if os.path.exists(snapshot_dir):
                    snapshot_hashes = os.listdir(snapshot_dir)
                    if snapshot_hashes:
                        # Use the first snapshot hash 
                        actual_model_path = os.path.join(snapshot_dir, snapshot_hashes[0])
                        logger.info(f"Using HF cache snapshot at: {actual_model_path}")
            
            # Configure quantization
            bnb_config = None
            torch_dtype = torch.bfloat16  # BF16 is often best for Gemma
            
            if self.quantization == "4bit":
                logger.info("Using 4-bit quantization")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True
                )
            elif self.quantization == "8bit":
                logger.info("Using 8-bit quantization")
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
            else:
                logger.info("Using full precision (no quantization)")
            
            # Load tokenizer from located path
            logger.info(f"Loading tokenizer from {actual_model_path}...")
            tokenizer = AutoTokenizer.from_pretrained(
                actual_model_path, 
                trust_remote_code=True,
                local_files_only=True  # Ensure we only use local files
            )
            logger.info("Tokenizer loaded successfully")
            
            # Prepare model loading arguments
            model_kwargs = {
                "quantization_config": bnb_config,
                "torch_dtype": torch_dtype if bnb_config is None else None,
                "device_map": self.device,
                "trust_remote_code": True,
                "local_files_only": True,  # Ensure we only use local files
            }
            
            # Add Flash Attention configuration if enabled
            if self.use_flash_attention:
                try:
                    # Check if flash_attn is installed
                    import flash_attn
                    logger.info("Flash Attention is available, will be used if hardware supports it")
                    
                    # Add Flash Attention config
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                except ImportError:
                    logger.warning("Flash Attention package not found. Falling back to standard attention.")
                    logger.warning("To use Flash Attention, install: pip install flash-attn")
            
            # Load model from located path
            logger.info(f"Loading model from {actual_model_path}...")
            model = AutoModelForCausalLM.from_pretrained(
                actual_model_path,
                **model_kwargs
            )
            logger.info("Model loaded successfully")
            
            # Set pad token if missing
            if tokenizer.pad_token is None:
                logger.warning("Tokenizer missing pad token; setting to eos_token")
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = model.config.eos_token_id
            
            model.eval()  # Set to evaluation mode
            
            # Store the model and tokenizer
            self.model = model
            self.tokenizer = tokenizer
            
            # Log device information
            if hasattr(model, "hf_device_map"):
                logger.info(f"Model loaded on device map: {model.hf_device_map}")
            else:
                device_info = next(model.parameters()).device
                logger.info(f"Model loaded on device: {device_info}")
            
            # Log attention implementation
            if hasattr(model.config, "attn_implementation"):
                logger.info(f"Using attention implementation: {model.config.attn_implementation}")
                
            return model, tokenizer
            
        except ImportError as e:
            logger.error(f"ImportError: {e}. Make sure 'transformers', 'torch', 'accelerate', and 'bitsandbytes' (if quantizing) are installed.")
            return None, None
        except OSError as e:
            logger.error(f"OSError: {e}. Model files might be missing or corrupted in {self.model_path}.")
            return None, None
        except Exception as e:
            logger.error(f"Unexpected error during model loading: {e}", exc_info=True)
            return None, None
    
    
    def unload_model(self):
        """
        Unload the model to free up memory.
        """
        if self.model is not None:
            try:
                del self.model
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                self.model = None
                logger.info("Model unloaded successfully")
            except Exception as e:
                logger.error(f"Error unloading model: {e}")
                
    def get_model_info(self):
        """
        Get information about the current model configuration.
        
        Returns:
            dict: Model configuration information
        """
        info = {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "model_size": self.model_size,
            "quantization": self.quantization,
            "device": self.device,
            "flash_attention": self.use_flash_attention,
            "is_loaded": self.model is not None
        }
        
        # Add additional model information if loaded
        if self.model is not None:
            if hasattr(self.model.config, "model_type"):
                info["model_type"] = self.model.config.model_type
                
            if hasattr(self.model.config, "attn_implementation"):
                info["attention_implementation"] = self.model.config.attn_implementation
            
            # Try to get memory usage info for CUDA devices
            try:
                if torch.cuda.is_available():
                    info["cuda_memory"] = {
                        "allocated": f"{torch.cuda.memory_allocated() / (1024**3):.2f} GB",
                        "reserved": f"{torch.cuda.memory_reserved() / (1024**3):.2f} GB", 
                        "max_allocated": f"{torch.cuda.max_memory_allocated() / (1024**3):.2f} GB"
                    }
            except:
                pass
        
        return info

    def is_loaded(self):
        """Check if model is loaded."""
        return self.model is not None and self.tokenizer is not None