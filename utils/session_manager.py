# utils/session_manager.py

import logging
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import random
import re # Import re for cleaning text

logger = logging.getLogger(__name__)

class InterviewSessionManager:
    """
    Manages interview session state, question flow, and answer tracking.
    Provides a central point for handling all session-related operations.
    """

    STATES = {
        'NOT_STARTED': 'not_started',
        'IN_PROGRESS': 'in_progress',
        'AWAITING_CONFIRMATION': 'awaiting_confirmation',
        'COMPLETED': 'completed',
        'ERROR': 'error'
    }

    def __init__(self, questions_data=None, randomize_order=False):
        """
        Initialize the session manager.

        Args:
            questions_data: List of question objects or path to questions JSON file
            randomize_order: Whether to randomize question order
        """
        self.questions = []
        self.questions_by_id = {}
        self.randomize_order = randomize_order

        if questions_data:
            if isinstance(questions_data, str):
                try:
                    with open(questions_data, 'r') as f:
                        loaded_questions = json.load(f)
                    logger.info(f"Loaded {len(loaded_questions)} questions from {questions_data}")
                    self.questions = loaded_questions
                except Exception as e:
                    logger.error(f"Failed to load questions from file: {e}", exc_info=True)
                    self.questions = [] # Ensure it's an empty list on error
            elif isinstance(questions_data, list):
                self.questions = questions_data
                logger.info(f"Using provided questions list with {len(self.questions)} items")
            else:
                 logger.error(f"Invalid questions_data type: {type(questions_data)}. Expected list or str path.")
                 self.questions = []

        # Build lookup dictionary by question ID (ensure IDs are integers)
        if self.questions:
            temp_lookup = {}
            processed_questions = []
            for q in self.questions:
                 try:
                      q_id_int = int(q['id'])
                      # Ensure 'text' key exists
                      if 'text' not in q:
                           logger.warning(f"Question ID {q_id_int} missing 'text' field. Skipping.")
                           continue
                      temp_lookup[q_id_int] = q
                      processed_questions.append(q) # Keep only valid questions
                 except (KeyError, ValueError, TypeError) as e:
                      logger.warning(f"Skipping question due to invalid ID/format: {q}. Error: {e}")

            self.questions = processed_questions # Update questions list
            self.questions_by_id = temp_lookup
            if not self.questions_by_id:
                 logger.error("No valid questions could be processed into questions_by_id lookup.")
            else:
                 logger.info(f"Built questions_by_id lookup with {len(self.questions_by_id)} valid items.")

        self.default_session_state_structure = {
            # ... (as before)
             'id': None, 'state': self.STATES['NOT_STARTED'], 'started_at': None,
             'last_updated': None, 'chat_history': [], 'current_question_id': None,
             'last_question_id': None, 'answered_questions': {},
             'remaining_question_ids': [], 'pending_answer': None,
             'completed_at': None
        }
        logger.debug("InterviewSessionManager initialized.")


    def create_session(self, session_id=None, randomize=None) -> Dict[str, Any]:
        """ Creates a new interview session. """
        if not self.questions_by_id: # Check the lookup dict
             logger.error("Cannot create session: No valid questions loaded into lookup.")
             return {'id': None, 'state': self.STATES['ERROR'], 'error': 'No valid questions loaded'}

        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"

        should_randomize = randomize if randomize is not None else self.randomize_order
        question_ids = list(self.questions_by_id.keys()) # Get IDs from the lookup

        if should_randomize:
            random.shuffle(question_ids)
            logger.info(f"Created randomized question order for session {session_id}")

        current_time = datetime.now().isoformat()
        session_state = {
            'id': session_id,
            'state': self.STATES['NOT_STARTED'],
            'started_at': current_time,
            'last_updated': current_time,
            'chat_history': [],
            'current_question_id': None,
            'last_question_id': None,
            'answered_questions': {},
            'remaining_question_ids': question_ids, # List of int IDs
            'pending_answer': None,
            'completed_at': None
        }

        logger.info(f"Created new session with ID: {session_id}")
        return session_state

    def start_interview(self, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """ Starts the interview, setting the first question. """
        if not session_state or 'state' not in session_state:
             logger.error("Invalid session state passed to start_interview.")
             return session_state

        if session_state['state'] != self.STATES['NOT_STARTED']:
            logger.warning(f"Cannot start interview for session {session_state.get('id', 'N/A')}: already in state {session_state['state']}")
            return session_state

        session_state['state'] = self.STATES['IN_PROGRESS']
        now_iso = datetime.now().isoformat()
        if not session_state.get('started_at'):
            session_state['started_at'] = now_iso
        session_state['last_updated'] = now_iso

        if session_state.get('remaining_question_ids'):
            first_qid = session_state['remaining_question_ids'][0]
            session_state['current_question_id'] = first_qid
            logger.info(f"Starting interview session {session_state.get('id', 'N/A')} with question ID: {first_qid}")
        else:
            logger.error(f"Cannot start interview session {session_state.get('id', 'N/A')}: no remaining questions available")
            session_state['state'] = self.STATES['ERROR']
            session_state['current_question_id'] = None

        return session_state


    def record_answer(
        self,
        session_state: Dict[str, Any],
        question_id: Union[int, str],
        answer_value: Union[int, str]
    ) -> Dict[str, Any]:
        """ Records an answer, advances state, updates question IDs. """
        if not session_state or 'state' not in session_state:
             logger.error("Invalid session state passed to record_answer.")
             return session_state

        try:
             q_id_int = int(question_id)
        except (ValueError, TypeError):
             logger.error(f"Received invalid non-integer question ID type: {question_id} ({type(question_id)})")
             session_state['state'] = self.STATES['ERROR']
             return session_state

        if q_id_int not in self.questions_by_id:
            logger.warning(f"Attempted to record answer for unknown question ID: {q_id_int} in session {session_state.get('id', 'N/A')}")
            return session_state

        is_valid_score = isinstance(answer_value, int) and (1 <= answer_value <= 5)
        is_valid_skip = isinstance(answer_value, str) and answer_value.lower() == 'skipped'
        if not (is_valid_score or is_valid_skip):
            logger.warning(f"Invalid answer value '{answer_value}' for QID {q_id_int}. Must be 1-5 or 'skipped'")
            return session_state

        session_state['answered_questions'][q_id_int] = answer_value
        logger.info(f"Recorded answer for question {q_id_int}: {answer_value} in session {session_state.get('id', 'N/A')}")

        try:
            if q_id_int in session_state.get('remaining_question_ids', []):
                session_state['remaining_question_ids'].remove(q_id_int)
                logger.debug(f"Removed QID {q_id_int} from remaining_question_ids.")
            else:
                 logger.warning(f"QID {q_id_int} not found in remaining_question_ids for removal (was it already answered?). State: {session_state['state']}")
        except Exception as e:
             logger.error(f"Error removing QID {q_id_int} from remaining: {e}", exc_info=True)
             session_state['state'] = self.STATES['ERROR']
             return session_state

        session_state['last_question_id'] = q_id_int
        session_state['pending_answer'] = None

        if session_state.get('remaining_question_ids'):
            next_qid = session_state['remaining_question_ids'][0]
            session_state['current_question_id'] = next_qid
            session_state['state'] = self.STATES['IN_PROGRESS']
            logger.info(f"Next question selected: {next_qid}. State set to IN_PROGRESS.")
        else:
            session_state['current_question_id'] = None
            session_state['state'] = self.STATES['COMPLETED']
            session_state['completed_at'] = datetime.now().isoformat()
            logger.info(f"Interview completed for session {session_state.get('id', 'N/A')}")

        session_state['last_updated'] = datetime.now().isoformat()
        return session_state

    def add_message_to_history(
        self,
        session_state: Dict[str, Any],
        role: str,
        content: str
    ) -> Dict[str, Any]:
        """ Adds a message to the chat history. """
        if not session_state or 'chat_history' not in session_state or not isinstance(session_state['chat_history'], list):
            logger.error(f"Invalid session state or chat_history for add_message_to_history in session {session_state.get('id', 'N/A')}")
            return session_state

        if role not in ['user', 'model']:
            logger.warning(f"Invalid message role: {role}. Must be 'user' or 'model'")
            return session_state

        session_state['chat_history'].append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        session_state['last_updated'] = datetime.now().isoformat()
        logger.debug(f"Added {role} message to history. New length: {len(session_state['chat_history'])}")
        return session_state

    def get_question_text_by_id(self, question_id: Optional[Union[int, str]]) -> Optional[str]:
        """
        Get the CORE text (statement part) of a question by its integer ID.
        Removes the "I see myself as someone who" prefix if present.
        """
        if question_id is None:
            return None
        try:
            q_id_int = int(question_id) # Ensure integer ID for lookup
        except (ValueError, TypeError):
            logger.warning(f"Invalid type for question_id lookup: {question_id} ({type(question_id)})")
            return None

        question_data = self.questions_by_id.get(q_id_int)
        if question_data:
            raw_text = question_data.get('text')
            if raw_text:
                # Remove the common prefix, case-insensitive, handling potential extra space
                core_text = re.sub(r"^i see myself as someone who\s*", "", raw_text, flags=re.IGNORECASE).strip()
                # Further cleanup (optional): remove trailing punctuation if needed
                # core_text = core_text.rstrip('.?!')
                logger.debug(f"Returning core text for QID {q_id_int}: '{core_text}' (from raw: '{raw_text}')")
                return core_text
            else:
                logger.warning(f"Question data found for ID {q_id_int}, but 'text' field is missing or empty.")
                return '[Question text missing]'
        else:
            logger.warning(f"Could not find question data for ID: {q_id_int}")
            return None
    # --- END OF MISSING METHOD ---


    def get_next_question_text(self, session_state: Dict[str, Any]) -> Optional[str]:
        """ DEPRECATED / Less Useful: Use get_question_text_by_id instead. """
        # Keeping for potential backward compatibility or other uses, but prefer the ID lookup.
        if not session_state: return None
        question_id = session_state.get('current_question_id')
        return self.get_question_text_by_id(question_id) # Delegate to the new method

    def get_interview_stats(self, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """ Get statistics about the interview progress. """
        default_stats = {
            'total_questions': 0, 'answered_count': 0, 'remaining_count': 0,
            'progress_percentage': 0, 'trait_coverage': {}, 'state': self.STATES['ERROR'],
            'started_at': None, 'last_updated': None, 'completed_at': None
        }
        if not session_state or not self.questions_by_id: # Check lookup
             logger.warning("Cannot get stats: Invalid session state or no questions loaded.")
             return default_stats

        total_questions = len(self.questions_by_id) # Use lookup count
        answered_questions_dict = session_state.get('answered_questions', {})
        remaining_question_ids_list = session_state.get('remaining_question_ids', [])

        answered_count = len(answered_questions_dict)
        remaining_count = len(remaining_question_ids_list)

        # Basic sanity check
        if total_questions > 0 and total_questions != answered_count + remaining_count:
             logger.warning(f"Stats count mismatch: Total({total_questions}) != Answered({answered_count}) + Remaining({remaining_count})")
             # Recalculate remaining just to be safe for percentage, though list length should be source of truth
             # remaining_count = total_questions - answered_count

        trait_coverage = {}
        trait_counts = {}
        for q_id, q_data in self.questions_by_id.items(): # Iterate lookup
            trait = q_data.get('trait')
            if trait:
                if trait not in trait_counts: trait_counts[trait] = {'total': 0, 'answered': 0}
                trait_counts[trait]['total'] += 1
                if q_id in answered_questions_dict: # Check int ID
                    trait_counts[trait]['answered'] += 1

        for trait, counts in trait_counts.items():
            if counts['total'] > 0:
                percentage = (counts['answered'] / counts['total']) * 100
                trait_coverage[trait] = {'answered': counts['answered'], 'total': counts['total'], 'percentage': round(percentage, 1)}

        progress_percentage = round((answered_count / total_questions) * 100, 1) if total_questions > 0 else 0

        return {
            'total_questions': total_questions,
            'answered_count': answered_count,
            'remaining_count': remaining_count,
            'progress_percentage': progress_percentage,
            'trait_coverage': trait_coverage,
            'state': session_state.get('state', self.STATES['ERROR']),
            'started_at': session_state.get('started_at'),
            'last_updated': session_state.get('last_updated'),
            'completed_at': session_state.get('completed_at')
        }


    def update_session_state(self, session_state: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """ Apply multiple updates to the session state. """
        if not session_state: return {}
        protected_fields = ['id', 'started_at', 'chat_history']
        for field, value in updates.items():
            if field in protected_fields:
                logger.warning(f"Attempted to directly update protected field: {field}")
                continue
            session_state[field] = value
        session_state['last_updated'] = datetime.now().isoformat()
        return session_state