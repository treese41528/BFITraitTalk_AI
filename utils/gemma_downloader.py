# llm/gemma_loader.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Gemma 3 Configuration ---
# DEFAULT_MODEL_NAME = "google/gemma-3-27b-it"
# DEFAULT_MODEL_NAME = "google/gemma-3-12b-it"
DEFAULT_MODEL_NAME = "google/gemma-3-4b-it"

# *** Specify your custom download/cache directory ***
# Make sure this path exists and you have write permissions
# CUSTOM_MODEL_CACHE_DIR = "data/hf_models/gemma-3-27b-it"
# CUSTOM_MODEL_CACHE_DIR = "data/hf_models/gemma-3-12b-it" 
CUSTOM_MODEL_CACHE_DIR = "data/hf_models/gemma-3-4b-it" 

# Quantization options: None, "8bit", "4bit"
# Start with None for A100, use "4bit" if facing memory issues or want more efficiency
DEFAULT_QUANTIZATION = None # Set to "4bit" or "8bit" if needed

# Specify device: "auto" is best to let accelerate handle (multiple) GPUs
DEVICE = "auto"

# Use bfloat16 on A100s for better performance/stability with large models
TORCH_DTYPE = torch.bfloat16

# Enable Flash Attention 2 if installed (pip install flash-attn)
USE_FLASH_ATTENTION_2 = True
# --- End Configuration ---


def load_model_and_tokenizer(model_name=None, quantization=None, device=DEVICE, cache_dir=CUSTOM_MODEL_CACHE_DIR):
    """
    Loads the Gemma 3 model and tokenizer with optional quantization,
    saving to/loading from a specific cache directory.

    Args:
        model_name (str, optional): Hugging Face model ID. Defaults to DEFAULT_MODEL_NAME.
        quantization (str, optional): Quantization mode ('4bit', '8bit', None). Defaults to DEFAULT_QUANTIZATION.
        device (str): Device to load the model onto ('auto', 'cuda', 'cpu'). Defaults to DEVICE.
        cache_dir (str): Path to the directory for downloading and caching the model. Defaults to CUSTOM_MODEL_CACHE_DIR.

    Returns:
        tuple: (model, tokenizer) or (None, None) if loading fails.
    """
    model_name = model_name or DEFAULT_MODEL_NAME
    quantization = quantization or DEFAULT_QUANTIZATION

    logging.info(f"Attempting to load model: {model_name}")
    logging.info(f"Target cache directory: {cache_dir}")
    logging.info(f"Quantization: {quantization}")
    logging.info(f"Device map: {device}")
    logging.info(f"Using dtype: {TORCH_DTYPE}")
    logging.info(f"Flash Attention 2 enabled: {USE_FLASH_ATTENTION_2}")


    # Ensure the custom cache directory exists
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError as e:
        logging.error(f"Could not create cache directory {cache_dir}: {e}")
        return None, None

    try:
        # Configure quantization if requested
        bnb_config = None
        if quantization == "4bit":
            logging.info("Using 4-bit quantization.")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=TORCH_DTYPE,
                bnb_4bit_use_double_quant=True,
            )
        elif quantization == "8bit":
            logging.info("Using 8-bit quantization.")
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        else:
            logging.info("No quantization applied.")

        # Determine attention implementation
        attn_implementation = "flash_attention_2" if USE_FLASH_ATTENTION_2 else "sdpa" # sdpa is fallback

        # Load tokenizer, specifying the cache directory
        logging.info(f"Loading tokenizer for {model_name} (cache: {cache_dir})...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        logging.info("Tokenizer loaded successfully.")

        # Load model, specifying cache_dir, dtype, quantization, device_map, attn
        logging.info(f"Loading model {model_name} (cache: {cache_dir})...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=TORCH_DTYPE if bnb_config is None else None, # dtype handled by bnb if quantizing
            device_map=device,
            cache_dir=cache_dir,
            trust_remote_code=True,
            attn_implementation=attn_implementation
        )
        logging.info("Model loaded successfully.")

        # Set pad token if missing
        if tokenizer.pad_token is None:
            logging.warning("Tokenizer missing pad token; setting to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        model.eval() # Set model to evaluation mode

        logging.info(f"Model {model_name} ready on device map: {model.hf_device_map}")
        return model, tokenizer

    except ImportError as e:
        logging.error(f"ImportError: {e}. Missing libraries? (transformers, torch, accelerate, flash-attn?, bitsandbytes?)")
        return None, None
    except OSError as e:
        # Can be permission error, disk space error, or model download issue
        logging.error(f"OSError: {e}. Check permissions for {cache_dir}, disk space, internet connection, and if you accepted the model license on Hugging Face.")
        return None, None
    except Exception as e:
        logging.error(f"An unexpected error occurred during model loading: {e}", exc_info=True)
        return None, None

if __name__ == "__main__":
    print("Testing model loading with custom cache directory...")
    model, tokenizer = load_model_and_tokenizer()
    if model and tokenizer:
        print(f"Model and Tokenizer loaded successfully! Model device map: {model.hf_device_map}")
        # Optional: Add a quick generation test here
        # try:
        #     prompt = [{"role": "user", "content": "Hello!"}]
        #     formatted_prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        #     inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        #     outputs = model.generate(**inputs, max_new_tokens=10)
        #     print("Test generation:", tokenizer.decode(outputs[0], skip_special_tokens=True))
        # except Exception as test_e:
        #     print(f"Test generation failed: {test_e}")
    else:
        print("Failed to load model and tokenizer.")