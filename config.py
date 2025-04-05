# config.py

import os

# --- Application Configuration ---

# Debugging mode
DEBUG_MODE = os.environ.get("DEBUG_MODE", "True").lower() in ("true", "1", "t")

# --- Model Configuration ---

# Path to the locally saved models
LOCAL_MODELS_PATH = os.environ.get("LOCAL_MODELS_PATH", "data/hf_models")

# Model size to use: '4b', '12b', or '27b'
MODEL_SIZE = os.environ.get("GEMMA_MODEL_SIZE", "12b")

# Quantization level: 'none', '4bit', or '8bit'
QUANTIZATION = os.environ.get("GEMMA_QUANTIZATION", "4bit")

# Device for model placement: 'auto', 'cuda', 'cpu'
DEVICE = os.environ.get("GEMMA_DEVICE", "auto")

# Whether to use Flash Attention if available
USE_FLASH_ATTENTION = os.environ.get("USE_FLASH_ATTENTION", "True").lower() in ("true", "1", "t")

# --- Model Generation Parameters ---

# Temperature for generation (higher = more creative)
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))

# Top-p (nucleus) sampling parameter
TOP_P = float(os.environ.get("TOP_P", "0.9"))

# Maximum new tokens to generate per response
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "1400"))

# --- Data Configuration ---

# Path to BFI questions data
BFI_QUESTIONS_PATH = os.environ.get("BFI_QUESTIONS_PATH", "data/bfi_items.json")

# --- Server Configuration ---

# Host to run the server on
HOST = os.environ.get("FLASK_HOST", "0.0.0.0")

# Port to run the server on
PORT = int(os.environ.get("FLASK_PORT", "5000"))

# Flask secret key (auto-generated if not provided)
SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", None)

# --- Logging Configuration ---

# Log level
LOG_LEVEL = os.environ.get("LOG_LEVEL", "DEBUG")

# Log format
LOG_FORMAT = os.environ.get("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# --- Available Models ---

# Dictionary mapping model sizes to model directory names
MODEL_NAMES = {
    "4b": "gemma-3-4b-it",
    "12b": "gemma-3-12b-it",
    "27b": "gemma-3-27b-it",
}

# Function to validate configuration
def validate_config():
    """
    Validate the configuration settings.
    
    Returns:
        tuple: (is_valid, message)
    """
    # Check if the models directory exists
    if not os.path.exists(LOCAL_MODELS_PATH):
        return False, f"Models directory not found: {LOCAL_MODELS_PATH}"
    
    # Check if the selected model exists
    model_name = MODEL_NAMES.get(MODEL_SIZE)
    if not model_name:
        return False, f"Invalid model size: {MODEL_SIZE}. Available options: {list(MODEL_NAMES.keys())}"
    
    model_path = os.path.join(LOCAL_MODELS_PATH, model_name)
    if not os.path.exists(model_path):
        return False, f"Selected model path not found: {model_path}"
    
    # Check if BFI questions file exists
    if not os.path.exists(BFI_QUESTIONS_PATH):
        return False, f"BFI questions file not found: {BFI_QUESTIONS_PATH}"
    
    # Validate numerical parameters
    try:
        assert 0.0 <= TEMPERATURE <= 2.0, "Temperature must be between 0.0 and 2.0"
        assert 0.0 < TOP_P <= 1.0, "Top-p must be between 0.0 and 1.0"
        assert 1 <= MAX_NEW_TOKENS <= 2048, "MAX_NEW_TOKENS must be between 1 and 2048"
    except AssertionError as e:
        return False, f"Invalid parameter: {str(e)}"
    
    # Check if Flash Attention is available if enabled
    if USE_FLASH_ATTENTION:
        try:
            import flash_attn
            flash_attention_status = "Available"
        except ImportError:
            flash_attention_status = "Not installed (will fall back to standard attention)"
    else:
        flash_attention_status = "Disabled by configuration"
    
    # All checks passed
    config_summary = [
        f"Model: {model_name}",
        f"Quantization: {QUANTIZATION}",
        f"Device: {DEVICE}",
        f"Flash Attention: {flash_attention_status}",
        f"Generation Parameters: temp={TEMPERATURE}, top_p={TOP_P}, max_tokens={MAX_NEW_TOKENS}"
    ]
    
    return True, "Configuration valid.\n" + "\n".join(config_summary)

# Function to get model parameters for generation
def get_generation_params():
    """
    Get the configured generation parameters.
    
    Returns:
        dict: Generation parameters
    """
    return {
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": TEMPERATURE > 0.0,
        "repetition_penalty": 1.1,
    }

# Print configuration summary if run directly
if __name__ == "__main__":
    valid, message = validate_config()
    if valid:
        print(f"✓ Configuration is valid:")
        print(message)
    else:
        print(f"✗ Configuration error: {message}")