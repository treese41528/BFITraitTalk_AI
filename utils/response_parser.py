# utils/response_parser.py

import re
import logging
from typing import Dict, Any, Optional, List

# Configure logging - Ensure this is configured ONLY in survey_app.py
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Get logger for this module

class ResponseParser:
    """
    Parses AI responses to extract answer confirmations and detect asked questions.
    """

    def __init__(self):
        """Initialize the parser with relevant patterns."""
        logger.debug("Initializing ResponseParser")
        # Patterns for detecting explicit score confirmations
        self.score_patterns = [
            # Made 'as a' optional, added 'recording'
            r'(?:mark|record|recording|put|rate|score|set)\s+(?:that|it|this|you)\s+(?:as\s+(?:a\s+)?|at\s+)?(\d)(?:\s+(?:out\s+of\s+5|on the scale|points))?',
            r'sounds\s+like\s+(?:a\s+)?(\d)(?:\s+to\s+me)?',
            r'put\s+you\s+down\s+as\s+(?:a\s+)?(\d)'
        ]
        # Patterns for detecting skips
        self.skip_patterns = [
            r'(?:we\s+(?:can|will)|I\'ll|I\s+will)\s+skip\s+(?:that|this|the\s+question|it)', # Added 'it'
            r'(?:let\'s|we\s+can)\s+move\s+(?:on|to\s+the\s+next)', # Made broader
            r'(?:mark|record)(?:ing|ed)?\s+(?:that|this|it)\s+as\s+skipped',
            r'no\s+problem\b.{0,25}\b(?:skip|mov(?:e|ing)|next)', # Slightly longer context window
            r'we\s+can\s+absolutely\s+(?:skip|move)',
            r'okay\s+to\s+skip'
        ]
        # Scale terms mapping for *implied* score (currently not used by extract_confirmed_answer)
        self.scale_terms = {
            'strongly agree': 5, 'agree': 4, 'neutral': 3, 'disagree': 2, 'strongly disagree': 1,
            'neither agree nor disagree': 3
        }
        # --- Revised BFI Question Detection Patterns ---
        # Pattern 1: Standard phrasing with "I see myself..."
        self.bfi_question_pattern_std = re.compile(
            r"(?:how\s+much\s+do\s+you\s+agree\s+with|thinking\s+about\s+yourself.*?):\s*['‘“]\s*I see myself as someone who\s*(.*?)\s*['’”]\??",
            re.IGNORECASE | re.DOTALL
        )
        # Pattern 2: Intro phrase directly followed by item text within quotes (AI might skip "I see myself...")
        self.bfi_question_pattern_direct = re.compile(
            r"(?:how\s+much\s+do\s+you\s+agree\s+with|thinking\s+about\s+yourself.*?):\s*['‘“]\s*(.*?)\s*['’”]\??",
            re.IGNORECASE | re.DOTALL
        )
        # Pattern 3: Just the "I see myself..." part in quotes (less reliable intro)
        self.bfi_question_pattern_quoted_core = re.compile(
             r"['‘“]\s*I see myself as someone who\s*(.*?)\s*['’”]\??",
             re.IGNORECASE | re.DOTALL
        )
        # Pattern 4: Just quoted text (least reliable, might catch other quotes)
        # self.bfi_question_pattern_quoted_only = re.compile(
        #      r"['‘“]\s*(.*?)\s*['’”]\??",
        #      re.IGNORECASE | re.DOTALL
        # )
        # --- End Revised Patterns ---

        logger.debug("ResponseParser patterns initialized.")



    def extract_confirmed_answer(self, text: str) -> Optional[Dict[str, Any]]:
        """ Extracts EXPLICIT score or skip confirmation from AI text. """
        # (Keep implementation as before - this parses the AI's proposed score)
        if not text: return None
        text_lower = text.lower()

        # 1. Check for EXPLICIT scores first
        for pattern in self.score_patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    score = int(match.group(1))
                    if 1 <= score <= 5:
                        logger.debug(f"Parser extracted explicit score: {score} via pattern '{pattern}'")
                        return {'value': score, 'type': 'score', 'match_text': match.group(0)}
                except (ValueError, IndexError): continue

        # 2. Check for EXPLICIT skips
        for pattern in self.skip_patterns:
            match = re.search(pattern, text_lower)
            if match:
                logger.debug(f"Parser detected explicit skip via pattern '{pattern}': {match.group(0)}")
                return {'value': 'skipped', 'type': 'skip', 'match_text': match.group(0)}

        logger.debug("No explicit score or skip confirmation found in AI text.")
        return None

    # --- FIX 5: Add user confirmation detection ---
    def check_user_confirmation(self, text: str) -> Optional[bool]:
        """
        Checks if user text indicates 'yes' or 'no' confirmation.

        Args:
            text: The user's response text.

        Returns:
            Optional[bool]: True for yes, False for no, None for unclear.
        """
        if not text:
            return None
        text_lower = text.lower().strip()

        # Define patterns for 'yes' and 'no'
        yes_patterns = [
            r"^\s*yes\b.*", r"^\s*yeah\b.*", r"^\s*yep\b.*", r"^\s*correct\b.*",
            r"^\s*that's right\b.*", r"^\s*sounds right\b.*", r"^\s*accurate\b.*",
            r"^\s*confirm\b.*", r"^\s*ok(ay)?\b.*", r"^\s*sure\b.*",
            # Specific positive responses to "Does that feel right?"
             r"^\s*it does\b.*"
        ]
        no_patterns = [
            r"^\s*no\b.*", r"^\s*nope\b.*", r"^\s*incorrect\b.*", r"^\s*wrong\b.*",
            r"^\s*that's not right\b.*", r"^\s*not really\b.*",
            # Specific negative responses like "no, make it a 3"
            r"^\s*no,.*(?:score|rate|mark|value|make it).*\d",
             r"^\s*actually,.*(?:score|rate|mark|value|make it).*\d",
        ]

        # Check for 'yes'
        for pattern in yes_patterns:
            if re.match(pattern, text_lower):
                # Avoid matching things like "no problem" as yes
                if "no problem" in text_lower and len(text_lower) < 15:
                     continue # Treat "no problem" alone as ambiguous or skip-related
                logger.debug(f"User confirmation detected: YES (pattern: '{pattern}')")
                return True

        # Check for 'no'
        for pattern in no_patterns:
            if re.match(pattern, text_lower):
                logger.debug(f"User confirmation detected: NO (pattern: '{pattern}')")
                return False

        # If neither yes nor no is clearly detected
        logger.debug("User confirmation unclear.")
        return None

    def detect_asked_question(self, ai_response_text: str, questions_data: List[Dict[str, Any]]) -> Optional[int]:
        """ Tries to identify which BFI question ID was asked in the AI's response. """
        # (Keep implementation as before - less critical now but useful for logging/debugging)
        if not ai_response_text or not questions_data:
            return None

        extracted_text = None
        # Try patterns in order of specificity
        match = self.bfi_question_pattern_std.search(ai_response_text) or \
                self.bfi_question_pattern_direct.search(ai_response_text) or \
                self.bfi_question_pattern_quoted_core.search(ai_response_text)

        if match:
            extracted_text = match.group(1).strip().lower()
            extracted_text = re.sub(r'^[.…,;:-]+|[.…,;:-]+$', '', extracted_text).strip()
            logger.debug(f"Parser trying to match extracted question text: '{extracted_text}'")

            if not extracted_text:
                 logger.debug("Extracted text was empty after cleaning.")
                 return None

            best_match_id = None

            # Exact Match First
            for question in questions_data:
                q_text_lower = question.get('text', '').lower()
                # Remove "I see myself as someone who" prefix if present in data for robust matching
                q_text_lower_core = re.sub(r"^i see myself as someone who\s*", "", q_text_lower).strip()
                q_id = question.get('id')
                if q_text_lower_core and q_id is not None:
                    if q_text_lower_core == extracted_text:
                         logger.info(f"Detected QID {q_id} via EXACT text match.")
                         return q_id # Return immediately on exact match

            # Substring Check Second
            possible_matches = []
            for question in questions_data:
                q_text_lower = question.get('text', '').lower()
                q_text_lower_core = re.sub(r"^i see myself as someone who\s*", "", q_text_lower).strip()
                q_id = question.get('id')
                if q_text_lower_core and q_id is not None and len(q_text_lower_core) > 3:
                    # Check containment both ways
                    if q_text_lower_core in extracted_text or extracted_text in q_text_lower_core:
                        # Prioritize longer matches or matches where extracted text contains the item text
                        overlap_len = 0
                        if q_text_lower_core in extracted_text: overlap_len = len(q_text_lower_core)
                        elif extracted_text in q_text_lower_core: overlap_len = len(extracted_text)

                        possible_matches.append({'id': q_id, 'len': overlap_len})
                        logger.debug(f"Found potential substring match for QID {q_id} (overlap {overlap_len})")


            if possible_matches:
                 # Select the match with the largest overlap length
                 best_match = max(possible_matches, key=lambda x: x['len'])
                 best_match_id = best_match['id']
                 logger.info(f"Detected QID {best_match_id} via BEST substring match (length {best_match['len']}).")
                 return best_match_id

        if not match:
             logger.debug("No BFI question patterns matched in AI response.")
        elif best_match_id is None:
             logger.debug(f"No BFI question text matched extracted '{extracted_text}' via substring.")

        return None


# --- Main block for testing ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    parser = ResponseParser()
    test_questions = [{"id": 1, "text": "is talkative", }, {"id": 2, "text": "tends to find fault with others",}, {"id": 3, "text": "does a thorough job", }, {"id": 43, "text": "is easily distracted", "reverse": True}]

    print("\n--- Testing Confirmation ---")
    resp1 = "Okay, recording a 4." ; print(f"'{resp1}' -> {parser.analyze_response(resp1, 10)}")
    resp2 = "We can skip that."; print(f"'{resp2}' -> {parser.analyze_response(resp2, 11)}")
    resp3 = "Sounds good. Next..."; print(f"'{resp3}' -> {parser.analyze_response(resp3, 12)}")

    print("\n--- Testing Question Detection ---")
    ai_resp1 = 'Thinking about yourself, how much do you agree with: "I see myself as someone who is talkative?"'
    print(f"'{ai_resp1}' -> Detected: {parser.detect_asked_question(ai_resp1, test_questions)}")
    ai_resp2 = 'How about: “I see myself as someone who tends to find fault with others”?'
    print(f"'{ai_resp2}' -> Detected: {parser.detect_asked_question(ai_resp2, test_questions)}")
    ai_resp3 = 'Okay, let’s move on. How much do you agree with: "I am easily distracted"?' # Missing "I see myself..."
    print(f"'{ai_resp3}' -> Detected: {parser.detect_asked_question(ai_resp3, test_questions)}") # Should detect 43
    ai_resp4 = "That's interesting. Why?"
    print(f"'{ai_resp4}' -> Detected: {parser.detect_asked_question(ai_resp4, test_questions)}")