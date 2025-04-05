# llm/gemma_handler.py

import torch
import logging
import re
from typing import List, Dict, Optional, Union, Tuple, Any

logger = logging.getLogger(__name__)

class GemmaConversationHandler:
    """
    Handles conversation generation with Gemma 3 model.
    """

    # --- Add generation_params to __init__ ---
    def __init__(self, model, tokenizer, default_generation_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the conversation handler.

        Args:
            model: The loaded Hugging Face model
            tokenizer: The loaded Hugging Face tokenizer
            default_generation_params (Optional[Dict[str, Any]]): Default parameters for model.generate()
        """
        self.model = model
        self.tokenizer = tokenizer

        # --- Store default generation params ---
        self.default_generation_params = default_generation_params if default_generation_params else {}
        # Set some basic defaults if none provided at all
        if not self.default_generation_params:
             self.default_generation_params = {
                 "max_new_tokens": 200,
                 "temperature": 0.7,
                 "top_p": 0.9,
                 "do_sample": True,
                 "repetition_penalty": 1.1
             }
             logger.warning(f"No default generation params provided, using basic defaults: {self.default_generation_params}")
        else:
             logger.info(f"Conversation handler initialized with default generation params: {self.default_generation_params}")


        # Verify we have a valid model and tokenizer
        if not self.model or not self.tokenizer:
            logger.error("Model or tokenizer is None. Cannot initialize conversation handler.")
            # Maybe raise an error here?
            raise ValueError("Model and Tokenizer must be provided to GemmaConversationHandler")
        else:
            logger.info("Conversation handler initialized successfully")

    # ... (format_conversation and _manually_format_conversation methods as before) ...
    def format_conversation(self, chat_history: List[Dict[str, str]]) -> Optional[str]: # Added Optional return type hint
        """
        Formats the chat history for Gemma 3 model.

        Args:
            chat_history: List of dictionaries with 'role' and 'content' keys
                          e.g., [{'role': 'user', 'content': '...'},
                                {'role': 'model', 'content': '...'}]

        Returns:
            Optional[str]: Formatted prompt string for the model, or None if formatting fails
        """
        if not self.tokenizer:
             logger.error("Tokenizer not available for formatting.")
             return None
        try:
            logger.debug(f"Formatting chat history (length {len(chat_history)}): {chat_history}")
            # Verify chat_history format
            if not isinstance(chat_history, list):
                 raise ValueError(f"chat_history must be a list, got {type(chat_history)}")
            for turn in chat_history:
                if not isinstance(turn, dict) or 'role' not in turn or 'content' not in turn:
                    raise ValueError(f"Invalid turn format: {turn}. Must be dict with 'role' and 'content'.")
                if turn['role'] not in ['user', 'model']:
                    raise ValueError(f"Invalid role: {turn['role']}. Must be 'user' or 'model'.")
            logger.warning(f"Formatting chat history (length {len(chat_history)}): {chat_history}")
            # Check if tokenizer has chat template method
            if hasattr(self.tokenizer, "apply_chat_template"):
                # Use built-in chat template (recommended approach)
                prompt = self.tokenizer.apply_chat_template(
                    chat_history,
                    tokenize=False,
                    add_generation_prompt=True # Adds the final '<start_of_turn>model\n'
                )
                logger.debug(f"Using tokenizer's apply_chat_template method.")
            else:
                # Manual formatting for Gemma 3 (fallback)
                logger.warning("Tokenizer does not have apply_chat_template method, using manual formatting")
                prompt = self._manually_format_conversation(chat_history)

            return prompt

        except Exception as e:
            logger.error(f"Error formatting conversation: {e}", exc_info=True)
            return None

    def _manually_format_conversation(self, chat_history: List[Dict[str, str]]) -> str:
        """ Fallback manual formatting """
        # Simple implementation, might need refinement based on exact model expectations
        prompt_str = ""
        for turn in chat_history:
             prompt_str += f"<start_of_turn>{turn['role']}\n{turn['content']}<end_of_turn>\n"
        prompt_str += "<start_of_turn>model\n"
        return prompt_str


    # --- Modify generate_response to use stored defaults ---
    def generate_response(
        self,
        chat_history: List[Dict[str, str]],
        # Remove individual params here if always using defaults or pass overrides as kwargs
        # Or keep them to allow overriding per call:
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
        **kwargs # Allow passing other generate() args
    ) -> str:
        """
        Generates a response from the model based on chat history.
        Uses stored default generation parameters, allowing overrides.

        Args:
            chat_history: List of conversation turns
            max_new_tokens (Optional[int]): Override default max_new_tokens.
            temperature (Optional[float]): Override default temperature.
            top_p (Optional[float]): Override default top_p.
            do_sample (Optional[bool]): Override default do_sample.
            repetition_penalty (Optional[float]): Override default repetition_penalty.
            **kwargs: Additional keyword arguments passed directly to model.generate().

        Returns:
            str: Generated response or error message
        """
        if not self.model or not self.tokenizer:
             logger.error("Cannot generate response: Model or Tokenizer not initialized.")
             return "Error: LLM components not ready."

        # Format the conversation into a prompt
        prompt = self.format_conversation(chat_history)
        if prompt is None:
            return "Error: Could not format conversation for the model."

        # --- Combine default and override parameters ---
        gen_params = self.default_generation_params.copy() # Start with defaults

        # Apply overrides if provided
        if max_new_tokens is not None: gen_params['max_new_tokens'] = max_new_tokens
        if temperature is not None: gen_params['temperature'] = temperature
        if top_p is not None: gen_params['top_p'] = top_p
        if do_sample is not None: gen_params['do_sample'] = do_sample
        if repetition_penalty is not None: gen_params['repetition_penalty'] = repetition_penalty

        # Add any other kwargs passed directly
        gen_params.update(kwargs)

        # Ensure essential params have some value
        gen_params.setdefault('max_new_tokens', 200)
        gen_params.setdefault('pad_token_id', self.tokenizer.eos_token_id)


        try:
            # Get the device from the model
            # Ensure model has parameters loaded, handle case where model might be empty
            if not list(self.model.parameters()):
                 logger.error("Model has no parameters loaded.")
                 return "Error: Model parameters not loaded."
            device = next(self.model.parameters()).device
            logger.info(f"Generating response using device: {device}")
            logger.debug(f"Generation parameters being used: {gen_params}")


            # Tokenize the prompt
            # add_special_tokens=False because apply_chat_template usually adds BOS/EOS etc. Check model docs if needed.
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False
            ).to(device)

            # Generate response
            logger.info("Generating response...")
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids, # Pass input_ids directly
                    # attention_mask=inputs.attention_mask, # Pass attention mask if needed/generated
                    **gen_params # Pass combined generation parameters
                )

            # Extract only the newly generated tokens, not the prompt
            input_length = inputs.input_ids.shape[1]
            # Handle potential edge case where output is shorter than input (unlikely but safe)
            if outputs[0].shape[0] > input_length:
                 new_tokens = outputs[0][input_length:]
            else:
                 logger.warning("Output sequence length is not greater than input length. Returning empty.")
                 new_tokens = []


            # Decode the response
            response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            logger.info(f"Generated {len(new_tokens)} tokens")

            # Clean up response
            response_text = self._clean_response(response_text)

            return response_text

        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return "I'm having difficulty generating a response. Let's try again."

    # ... (_clean_response, extract_confirmed_answer, detect_next_question methods as before) ...
    def _clean_response(self, text: str) -> str:
        """ Clean up the generated response. """
        text = text.strip()
        # Basic cleanup, may need refinement based on observed model outputs
        text = text.replace("<|end_of_turn|>", "").replace("<|start_of_turn|>", "").strip() # Handle potential variations
        text = text.replace("<end_of_turn>", "").replace("<start_of_turn>", "").strip()

        # Remove incomplete sentences at the end if needed, or other artifacts
        # Example: if text.endswith("..."): text = text[:-3]

        if not text:
            return "(The AI seems to have generated an empty response. Let's try again.)"
        return text

    def extract_confirmed_answer(self, response_text: str, expected_question_id: int) -> Optional[Dict[str, Any]]: # Added expected_question_id
        """
        Extracts answer confirmations from the generated response for a specific question.

        Args:
            response_text: The text to analyze
            expected_question_id: The ID of the question we expect confirmation for.

        Returns:
            Optional[Dict]: Containing 'question_id', 'value' (1-5 or 'skipped'), 'type', 'match_text' if found, else None
        """
        if not response_text:
            return None

        text_lower = response_text.lower()
        confirmation = None

        # Pattern 1: Explicit score
        score_pattern = r'(?:mark|record|put|rate|score|set)\s+(?:that|it|this|you)\s+(?:as\s+(?:a\s+)?|at\s+)(\d)(?:\s+(?:out\s+of\s+5|on the scale|points))?'
        score_match = re.search(score_pattern, text_lower)
        if score_match:
            try:
                score = int(score_match.group(1))
                if 1 <= score <= 5:
                    logger.info(f"Extracted score: {score} via regex")
                    confirmation = {'value': score, 'type': 'score', 'match_text': score_match.group(0)}
            except (ValueError, IndexError): pass

        # Pattern 2: Explicit Skip
        if confirmation is None:
             skip_patterns = [
                r'(?:we\s+(?:can|will)|I\'ll|I\s+will)\s+skip\s+(?:that|this|the\s+question)',
                r'(?:let\'s|we\s+can)\s+move\s+(?:on|to\s+the\s+next)', # Removed "question" to be broader
                r'(?:mark|record)(?:ing|ed)?\s+(?:that|this|it)\s+as\s+skipped'
             ]
             for pattern in skip_patterns:
                 skip_match = re.search(pattern, text_lower)
                 if skip_match:
                     logger.info(f"Detected answer skip via regex: {skip_match.group(0)}")
                     confirmation = {'value': 'skipped', 'type': 'skip', 'match_text': skip_match.group(0)}
                     break # Found skip, no need to check other skip patterns

        # Pattern 3: Implied Score (Use cautiously)
        if confirmation is None:
            scale_patterns = {'strongly agree': 5, 'agree': 4, 'neutral': 3, 'disagree': 2, 'strongly disagree': 1}
            # Look for phrases like "sounds like agree", "that's neutral", "you are strongly disagreeing" etc.
            # Make patterns more robust if needed
            for term, value in scale_patterns.items():
                 # Added variations like "that is", "you are", optional "a"
                 confirmation_pattern = rf"(?:sounds like|that(?:'s| is)|you(?:'re| are))\s+(?:a |an )?\b{term}\b"
                 term_match = re.search(confirmation_pattern, text_lower)
                 if term_match:
                     logger.info(f"Extracted implied score from '{term}': {value} via regex")
                     confirmation = {'value': value, 'type': 'implied_score', 'match_text': term_match.group(0)}
                     break # Found one implied score

        if confirmation:
             confirmation['question_id'] = expected_question_id # Add the question ID
             return confirmation
        else:
             logger.debug(f"No answer confirmation found for QID {expected_question_id} in response.")
             return None


    def detect_next_question(self, response_text: str) -> bool:
        """ Detects if the response includes a new question. """
        if not response_text: return False
        text_lower = response_text.lower()
        # More robust checks might involve looking for BFI keywords + question mark
        # Or checking if the last sentence ends with '?' and isn't just a clarification ("Does that sound right?")
        if '?' in text_lower and not re.search(r'(?:sound right|is that correct|okay)\?$', text_lower):
            # Simple check, assumes questions aren't just yes/no confirmations
            logger.debug("Detected '?' potentially indicating a next question.")
            return True

        # Look for common question starters, avoiding simple confirmations
        question_starters = [ "how much do you", "thinking about", "next question", "let's move on to", "how about", "do you see yourself" ]
        for starter in question_starters:
             if starter in text_lower:
                  logger.debug(f"Detected potential question starter: '{starter}'")
                  return True
        return False