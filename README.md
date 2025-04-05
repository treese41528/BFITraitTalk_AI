# Conversational BFI Interviewer with Gemma 3

A Flask-based web application for conducting personality assessments using the Big Five Inventory (BFI) through a conversational interface powered by Google's Gemma 3 large language model.

## Overview

This application provides a conversational approach to completing the Big Five Inventory personality assessment. Instead of filling out a traditional form, users engage in a natural conversation with an AI interviewer that adapts questions, provides active listening, and guides the user through the assessment process.

Key features:
- Split-screen interface with chat and dynamic questionnaire
- Conversational administration of the BFI questionnaire
- Real-time form updates as questions are answered
- Comprehensive personality profile generation
- Local model execution for privacy and data control

## Technical Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **LLM**: Google Gemma 3 (4B, 12B, or 27B parameter version)
- **Libraries**: Transformers, PyTorch, BitsAndBytes

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended, CPU is possible but slow)
- Local Gemma 3 model files

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/conversational-bfi-interviewer.git
cd conversational-bfi-interviewer
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Local Models

Ensure your Gemma 3 models are correctly placed in the project structure:

```
data/hf_models/
├── gemma-3-4b-it/     # 4B parameter model
├── gemma-3-12b-it/    # 12B parameter model
└── gemma-3-27b-it/    # 27B parameter model
```

You only need one of these models to run the application. The 4B model is recommended for systems with limited GPU memory.

### Step 5: Configure the Application

Edit `config.py` to set the appropriate model size and quantization level based on your hardware:

```python
# For systems with limited GPU memory (4-8GB VRAM)
MODEL_SIZE = "4b"
QUANTIZATION = "4bit"

# For systems with more GPU memory (16+ GB VRAM)
# MODEL_SIZE = "12b"
# QUANTIZATION = "8bit"
```

## Running the Application

1. Start the Flask server:

```bash
python app.py
```

2. Open your web browser and navigate to:

```
http://localhost:5000
```

3. Click the "Start Interview" button to begin the personality assessment.

## Usage

1. The application presents a split-screen interface with a chat window on the left and the BFI questionnaire form on the right.

2. Start the interview by clicking the "Start" button. The AI interviewer will introduce itself and begin asking questions.

3. Answer each question naturally in the chat. The AI will adapt its follow-up questions based on your responses.

4. As you answer questions, the form on the right will be automatically updated.

5. You can also directly interact with the form by clicking on the Likert scale options (1-5).

6. After completing all questions, you'll receive a comprehensive personality profile based on your responses.

## Configuration Options

You can customize the application behavior using environment variables:

```bash
# Model Configuration
export GEMMA_MODEL_SIZE="4b"  # Options: "4b", "12b", "27b"
export GEMMA_QUANTIZATION="4bit"  # Options: "none", "4bit", "8bit"
export GEMMA_DEVICE="auto"  # Options: "auto", "cuda", "cpu"

# Server Configuration
export FLASK_HOST="0.0.0.0"
export FLASK_PORT=5000
export DEBUG_MODE=False

# Logging Configuration
export LOG_LEVEL="INFO"
```

## Hardware Requirements

Different model sizes and quantization levels have different hardware requirements:

| Model Size | Quantization | Minimum VRAM | Recommended VRAM |
|------------|--------------|--------------|------------------|
| 4B         | 4-bit        | 2 GB         | 4 GB             |
| 4B         | 8-bit        | 4 GB         | 6 GB             |
| 12B        | 4-bit        | 6 GB         | 8 GB             |
| 12B        | 8-bit        | 12 GB        | 16 GB            |
| 27B        | 4-bit        | 14 GB        | 16 GB            |
| 27B        | 8-bit        | 27 GB        | 32 GB            |

The application can also run on CPU, but response times will be significantly slower.

## Troubleshooting

### "Out of memory" error

If you encounter CUDA out of memory errors:
- Try using a smaller model (e.g., 4B instead of 12B)
- Use more aggressive quantization (4-bit instead of 8-bit)
- Reduce the maximum token length in the configuration

### Model loading errors

If the model fails to load:
- Check that the model files are correctly placed in the expected directory
- Ensure you have sufficient disk space for the model files
- Verify that you have the correct permissions to access the model files

### Slow responses

If the AI is responding slowly:
- Try using a smaller model size
- Ensure you're using GPU acceleration if available
- Consider more aggressive quantization

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Big Five Inventory (BFI) is a widely used personality assessment tool
- Google's Gemma 3 models provide the conversational capabilities
- This project is for educational purposes only and not intended for clinical use