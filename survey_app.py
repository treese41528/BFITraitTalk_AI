# survey_app.py

# --- (Keep imports, logger setup, global vars, INITIAL_INTERVIEW_PROMPT,
#      load_bfi_questions, initialize_components, create_app, index, start_interview) ---
import re
from flask import Flask, render_template, request, jsonify, session, current_app
import json
import os
import logging
import secrets
from datetime import datetime
from flask_session import Session

import config
from llm.gemma_loader import GemmaModelManager
from llm.gemma_handler import GemmaConversationHandler
from utils.session_manager import InterviewSessionManager
from utils.response_parser import ResponseParser
from utils.bfi_scoring import BFIScorer

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL, 'INFO'), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

model_manager = None
conversation_handler = None
response_parser = ResponseParser()
session_manager = None
bfi_scorer = None
bfi_questions = None

INITIAL_INTERVIEW_PROMPT = """
You are 'Kaya', a friendly and professional AI interviewer conducting a personality assessment using the Big Five Inventory (BFI).
Your goal is to guide the user through the questionnaire conversationally.

Instructions:
1. Introduce yourself briefly and explain the process (conversational survey, answer in own words).
2. Ask one BFI question at a time. Phrase the questions naturally, starting with "Thinking about yourself, how much do you agree with: 'I see myself as someone who... [statement text]?'" or similar conversational phrasing.
3. Wait for the user's response.
4. Practice active listening: Briefly acknowledge or summarize the user's answer (e.g., "Okay, so you feel...", "Got it, thanks for sharing.").
5. **Crucially**: After acknowledging, try to map their response to the 5-point Likert scale (1: Strongly Disagree, 2: Disagree, 3: Neutral, 4: Agree, 5: Strongly Agree). You **MUST state the score you are recording clearly using phrases like 'I will mark that as a [score]' or 'Okay, recording a [score] for that one.'** before asking if it feels right. For example: "Based on what you said, **I will mark that as a 4** out of 5 for Agree. Does that feel right?" or "Okay, **recording a 1** for Strongly Disagree."
6. **Wait for confirmation**: After stating the score, explicitly ask the user if that score is correct (e.g., "Does that feel right?", "Is that accurate?"). Do NOT ask the next BFI question until the user confirms the score for the current one (e.g., responds with 'yes', 'correct', etc.).
7. If the user corrects the score (e.g., "no, make it a 3"), acknowledge the correction and state the *new* score you are recording (e.g., "My apologies, recording a 3 then.") and proceed to the next question.
8. If the user asks to skip a question, acknowledge it supportively ("Okay, no problem, we can skip that one.") and record it as 'skipped', then ask the next question.
9. Maintain a warm, empathetic, and non-judgmental tone throughout.
10. After the user confirms a score/skip, seamlessly transition to the *next logical question* from the BFI sequence.
11. When all questions are done, provide a brief closing statement thanking the user.

Let's begin the interview now. Please start with your introduction and the first question from the inventory.
""" # Refined prompt slightly

def load_bfi_questions():
    """Load BFI questions from file, ensuring integer IDs."""
    # ... (Implementation as before) ...
    fallback_questions = [{"id": 1, "text": "is talkative", "trait": "Extraversion", "reverse": False},{"id": 2, "text": "tends to find fault with others", "trait": "Agreeableness", "reverse": True},{"id": 3, "text": "does a thorough job", "trait": "Conscientiousness", "reverse": False},{"id": 4, "text": "is depressed, blue", "trait": "Neuroticism", "reverse": False},{"id": 5, "text": "is original, comes up with new ideas", "trait": "Openness", "reverse": False}]
    try:
        with open(config.BFI_QUESTIONS_PATH, 'r') as f: questions = json.load(f)
        logger.info(f"Loaded {len(questions)} BFI questions from {config.BFI_QUESTIONS_PATH}")
        processed_questions = []
        for q in questions:
            try: q['id'] = int(q['id']); processed_questions.append(q)
            except (KeyError, ValueError, TypeError) as e: logger.warning(f"Skipping question due to invalid ID/format: {q}. Error: {e}")
        if not processed_questions: raise ValueError("No valid questions found.")
        logger.info(f"Processed {len(processed_questions)} questions with integer IDs.")
        return processed_questions
    except Exception as e: logger.error(f"Error loading/processing BFI questions: {e}", exc_info=True); logger.warning("Returning fallback questions."); return fallback_questions


def initialize_components(app_instance):
    """Initialize all global components."""
    # ... (Implementation as before) ...
    global model_manager, conversation_handler, session_manager, bfi_scorer, bfi_questions, response_parser
    is_valid, message = config.validate_config();
    if not is_valid: logger.error(f"Config validation failed: {message}"); return False
    logger.info(f"Configuration validated:\n{message}")
    bfi_questions = load_bfi_questions()
    if not bfi_questions: logger.error("Failed to load BFI questions."); return False
    app_instance.config['BFI_QUESTIONS'] = bfi_questions
    try: session_manager = InterviewSessionManager(questions_data=bfi_questions); logger.info("Session Manager initialized.")
    except Exception as e: logger.error(f"Failed to init Session Manager: {e}", exc_info=True); return False
    try: bfi_scorer = BFIScorer(bfi_questions); logger.info("BFI Scorer initialized.")
    except Exception as e: logger.error(f"Failed to init BFI Scorer: {e}", exc_info=True); return False
    logger.info("Response Parser initialized.")
    try:
        logger.info(f"Initializing Gemma model (size={config.MODEL_SIZE}, quantization={config.QUANTIZATION})")
        model_manager = GemmaModelManager(model_size=config.MODEL_SIZE, quantization=config.QUANTIZATION, device=config.DEVICE, use_flash_attention=config.USE_FLASH_ATTENTION)
        model, tokenizer = model_manager.load_model_and_tokenizer()
        if model and tokenizer:
            generation_params = config.get_generation_params()
            conversation_handler = GemmaConversationHandler(model, tokenizer, generation_params); logger.info("Gemma Handler initialized.")
            return True
        else: logger.error("Failed to load model/tokenizer from manager"); return False
    except Exception as e: logger.error(f"Error initializing LLM components: {e}", exc_info=True); return False


# --- App Factory Function ---
def create_app():
    """Creates and configures the Flask application."""
    # ... (Implementation as previously corrected, including Flask-Session setup) ...
    app = Flask(__name__)
    app.secret_key = config.SECRET_KEY if config.SECRET_KEY else secrets.token_hex(16)
    if not config.SECRET_KEY: logger.warning("Flask SECRET_KEY not set in config...")
    app.config["SESSION_TYPE"] = "filesystem"; app.config["SESSION_PERMANENT"] = False; app.config["SESSION_USE_SIGNER"] = True
    app.config["SESSION_FILE_DIR"] = os.path.join(os.path.abspath(os.path.dirname(__file__)), '.flask_session')
    try: os.makedirs(app.config["SESSION_FILE_DIR"], exist_ok=True); logger.info(f"Flask-Session using dir: {app.config['SESSION_FILE_DIR']}")
    except OSError as e: logger.error(f"Could not create Flask-Session dir: {e}.")
    Session(app)
    app_log_level = getattr(logging, config.LOG_LEVEL, logging.INFO); app.logger.setLevel(app_log_level); logging.getLogger().setLevel(app_log_level)
    logger.info(f"Flask app logger level set to: {config.LOG_LEVEL}")
    with app.app_context():
        logger.info("Initializing application components..."); success = initialize_components(app)
        if not success: logger.critical("FATAL: Component initialization failed."); raise RuntimeError("Component init failed.")
        else: logger.info("Application components initialized successfully.")

    # --- Routes ---
    @app.route('/')
    def index():
        # ... (Implementation as previously corrected) ...
        if not bfi_questions or not session_manager: return "Error: App not configured.", 500
        if 'session_state' not in session or not session.get('session_state'): session_state = session_manager.create_session(); session['session_state'] = session_state; sid = getattr(session, 'sid', 'N/A'); logger.info(f"Created new session: {sid}")
        else: session_state = session['session_state']; sid = getattr(session, 'sid', 'N/A'); logger.debug(f"Existing session: {sid}, State: {session_state.get('state', 'Unknown')}")
        state_value = session_state.get('state', 'error') if isinstance(session_state, dict) else 'error'
        return render_template('index.html', questions=bfi_questions, initial_state=state_value)


    def looks_like_direct_answer(text: str) -> bool:
        """
        Simple check if user input looks like a direct score or simple rating.
        Returns True if it looks like an answer, False otherwise.
        """
        text_lower = text.strip().lower()
        # Check for single digits 1-5
        if re.fullmatch(r"[1-5]", text_lower):
            return True
        # Check for simple Likert terms (adjust as needed)
        likert_terms = ["strongly agree", "agree", "neutral", "disagree", "strongly disagree",
                        "yes", "no", "skip", "maybe", "sometimes"] # Add terms user might use as answers
        if text_lower in likert_terms:
            return True
        # Check for very short responses that might imply rating
        if len(text_lower.split()) <= 3 and any(term in text_lower for term in ["agree", "disagree", "neutral", "skip"]):
            return True

        # Otherwise, assume it's more conversational or a question
        return False

    # --- start_interview needs no changes related to this error ---
    @app.route('/api/start', methods=['POST'])
    def start_interview():
        # (Keep implementation from previous fix - it seemed correct)
        if not session_manager or not conversation_handler: return jsonify({'success': False, 'error': 'App not ready'}), 500
        session_state = session_manager.create_session(); sid = getattr(session, 'sid', 'N/A'); logger.info(f"Starting new interview via /api/start for {sid}.")
        session_state = session_manager.start_interview(session_state)

        if session_state['state'] == session_manager.STATES['ERROR']:
            logger.error(f"Failed to start interview (state is ERROR) for session {sid}.")
            session['session_state'] = session_state
            return jsonify({'success': False, 'error': 'Failed to initialize interview questions.'}), 500

        first_qid = session_state.get('current_question_id')
        first_question_text_core = session_manager.get_question_text_by_id(first_qid)

        if not first_question_text_core:
            logger.error(f"Failed to get text for first question ID: {first_qid}")
            session_state['state'] = session_manager.STATES['ERROR']
            session['session_state'] = session_state
            return jsonify({'success': False, 'error': 'Failed to load first question text.'}), 500

        first_question_full = f"I see myself as someone who {first_question_text_core}"
        initial_prompt_with_q = f"""
        {INITIAL_INTERVIEW_PROMPT}

        Let's begin. Thinking about yourself, how much do you agree with: '{first_question_full}'?
        """
        initial_history_for_ai = [{"role": "user", "content": initial_prompt_with_q}]

        try:
            logger.info(f"Generating initial AI response for {sid} (asking QID {first_qid})"); generation_params = config.get_generation_params();
            ai_response = conversation_handler.generate_response(initial_history_for_ai, **generation_params)
            if not ai_response or not ai_response.strip(): logger.error("LLM empty initial response."); raise ValueError("LLM empty response")
            logger.info(f"Initial AI response generated (len {len(ai_response)} chars)")

            # Add the prompt we sent and the AI's response to the actual history
            session_state = session_manager.add_message_to_history(session_state, 'user', initial_prompt_with_q)
            session_state = session_manager.add_message_to_history(session_state, 'model', ai_response)

            session['session_state'] = session_state;
            logger.info(f"Session {sid} started. State: {session_state['state']}, Current QID: {session_state['current_question_id']}")

            return jsonify({
                'success': True,
                'ai_message': ai_response,
                'interview_state': session_state['state'],
                'current_question_id': session_state.get('current_question_id')
            })
        except Exception as e: logger.error(f"Error in /api/start generation: {e}", exc_info=True); session.pop('session_state', None); return jsonify({'success': False, 'error': 'Failed to generate initial AI response'}), 500




    # --- process_chat (Revised IN_PROGRESS and AWAITING_CONFIRMATION Logic) ---
    @app.route('/api/chat', methods=['POST'])
    def process_chat():
        """Process user message based on interview state with more conversational handling."""
        # (Initial checks for components, user_message, session_state remain the same)
        if not all([session_manager, conversation_handler, response_parser, bfi_questions]):
            logger.error("Chat components missing.");
            return jsonify({'success': False, 'error': 'App components not ready'}), 500

        data = request.json
        user_message = data.get('message', '').strip()
        if not user_message:
            logger.warning("Empty chat message received.")
            return jsonify({'error': 'No message provided'}), 400

        if 'session_state' not in session or not isinstance(session.get('session_state'), dict):
            logger.warning("No active/valid session for chat.")
            return jsonify({'error': 'No active interview session. Please refresh.'}), 400

        session_state = session['session_state']
        sid = getattr(session, 'sid', 'N/A')
        current_internal_state = session_state.get('state', 'Unknown')
        current_qid = session_state.get('current_question_id')
        last_qid = session_state.get('last_question_id')
        pending_answer = session_state.get('pending_answer')

        logger.info(f"--- Processing chat for {sid} ---")
        logger.info(f"State: {current_internal_state}, Current QID: {current_qid}, Last QID: {last_qid}, Pending: {pending_answer}")
        logger.info(f"User Message: '{user_message}'")

        form_update = None
        ai_response_text = "An internal error occurred."
        needs_generation = True # Assume we need generation

        try:
            # --- Add User Message to History ---
            session_state = session_manager.add_message_to_history(session_state, 'user', user_message)
            history_for_generation = session_state.get('chat_history', [])[:]

            # --- State Machine Logic ---
            if current_internal_state == session_manager.STATES['AWAITING_CONFIRMATION']:
                logger.debug("State is AWAITING_CONFIRMATION. Analyzing user response for yes/no.")
                user_confirmation = response_parser.check_user_confirmation(user_message)
                confirmation_qid = last_qid

                # --- Handle missing confirmation_qid or pending_answer (same as before) ---
                if confirmation_qid is None:
                    logger.error("State AWAITING_CONFIRMATION, but last_question_id is None.")
                    ai_response_text = "Sorry, error tracking question. Let's try again."
                    session_state['state'] = session_manager.STATES['ERROR']
                    needs_generation = False
                elif pending_answer is None:
                    logger.error(f"State AWAITING_CONFIRMATION QID {confirmation_qid}, but pending_answer is None.")
                    ai_response_text = f"Sorry, forgot suggested score for '{session_manager.get_question_text_by_id(confirmation_qid)}'. Could you restate?"
                    session_state['state'] = session_manager.STATES['IN_PROGRESS']
                    session_state['current_question_id'] = confirmation_qid
                    session_state['pending_answer'] = None
                    needs_generation = False
                # --- End Handle missing ---

                elif user_confirmation is True:
                    # --- User confirmed: Record and ask next (same as before) ---
                    logger.info(f"User CONFIRMED pending answer {pending_answer['value']} for QID {confirmation_qid}.")
                    session_state = session_manager.record_answer(session_state, confirmation_qid, pending_answer['value'])
                    form_update = {'question_id': confirmation_qid, 'answer': pending_answer['value']}
                    if session_state['state'] == session_manager.STATES['COMPLETED']:
                        ai_response_text = "Great, got it. That completes the questionnaire! Thank you."
                        needs_generation = False
                    elif session_state['state'] == session_manager.STATES['IN_PROGRESS']:
                        next_qid = session_state.get('current_question_id')
                        next_question_text_core = session_manager.get_question_text_by_id(next_qid)
                        if next_question_text_core:
                            next_question_full = f"I see myself as someone who {next_question_text_core}"
                            instruction = f"Great, thanks for confirming. Now, let's move to the next one. Thinking about yourself, how much do you agree with: '{next_question_full}'?"
                            logger.info(f"Instructing AI to ask next question (QID {next_qid})")
                            if history_for_generation: history_for_generation[-1]['content'] = f"{history_for_generation[-1]['content']}\n\n[System instruction: Ask the next question naturally using this phrasing: '{instruction}']"
                            ai_response_text = conversation_handler.generate_response(history_for_generation, **config.get_generation_params())
                            needs_generation = False
                        else: # Error getting next question text
                            logger.error(f"Confirmed answer, but failed to get text for next QID {next_qid}.")
                            ai_response_text = "Okay, got it. Apologies, error finding next question."
                            session_state['state'] = session_manager.STATES['ERROR']
                            needs_generation = False
                    else: # Error state from record_answer
                        ai_response_text = "Sorry, error recording answer."
                        needs_generation = False
                    # --- End User confirmed ---

                elif user_confirmation is False:
                    # --- User declined: Ask for correction (same as before) ---
                    logger.info(f"User DECLINED pending answer {pending_answer} for QID {confirmation_qid}.")
                    question_text_core = session_manager.get_question_text_by_id(confirmation_qid)
                    instruction = f"My apologies. What score (1-5) should I record for '{question_text_core}' instead? Or 'skip'?"
                    if history_for_generation: history_for_generation[-1]['content'] = f"{history_for_generation[-1]['content']}\n\n[System instruction: Ask for correct 1-5 score or skip. Instruction: '{instruction}']"
                    ai_response_text = conversation_handler.generate_response(history_for_generation, **config.get_generation_params())
                    session_state['state'] = session_manager.STATES['IN_PROGRESS']
                    session_state['current_question_id'] = confirmation_qid
                    session_state['pending_answer'] = None
                    needs_generation = False
                    # --- End User declined ---

                else: # Unclear confirmation - User said something other than yes/no
                    logger.info("User confirmation unclear. Acknowledging input and gently redirecting.")
                    # *** NEW LOGIC FOR UNCLEAR CONFIRMATION ***
                    # Instead of just repeating, acknowledge and guide back.
                    instruction = f"""
                    I understand you might have more to say or questions, and I appreciate that.
                    However, to keep us on track with the questionnaire for now, I just need to confirm the score for the last statement about '{session_manager.get_question_text_by_id(confirmation_qid)}'.
                    The score I suggested was {pending_answer['value']}.
                    Could you please first tell me if that score feels generally right ('yes') or wrong ('no')?
                    Or, if you prefer, tell me the score (1-5) you'd like me to record, or say 'skip'.
                    We can potentially discuss things further once the main questions are done.
                    """
                    if history_for_generation: history_for_generation[-1]['content'] = f"{history_for_generation[-1]['content']}\n\n[System instruction: Acknowledge user's last message ('{user_message}') briefly. Then, gently guide them back to confirming the proposed score ({pending_answer['value']}) with yes/no, providing a corrected score (1-5), or skipping. Use this guidance: '{instruction}']"
                    ai_response_text = conversation_handler.generate_response(
                        history_for_generation,
                        **config.get_generation_params()
                    )
                    # Remain in AWAITING_CONFIRMATION state, keep pending_answer
                    needs_generation = False
                    # *** END NEW LOGIC ***

            elif current_internal_state == session_manager.STATES['IN_PROGRESS']:
                logger.debug(f"State is IN_PROGRESS. Analyzing user response for QID {current_qid}.")
                target_qid = current_qid
                question_text_core = session_manager.get_question_text_by_id(target_qid)

                if target_qid is None or question_text_core is None:
                    # (Error handling as before)
                    logger.error("State IN_PROGRESS, but current_qid or text missing.")
                    ai_response_text = "Apologies, lost my place."
                    session_state['state'] = session_manager.STATES['ERROR']
                    needs_generation = False
                else:
                    question_text_full = f"I see myself as someone who {question_text_core}"

                    # *** NEW: Check if user response looks like a direct answer ***
                    if looks_like_direct_answer(user_message):
                        logger.debug(f"User message '{user_message}' looks like a direct answer. Proceeding to propose score.")
                        # --- Proceed with score proposal logic (same as before) ---
                        instruction = f"""
                        Based on the user's last message ('{user_message}'), please:
                        1. Briefly acknowledge their response naturally (e.g., "Okay, got it.").
                        2. Analyze their sentiment regarding the statement: '{question_text_full}'.
                        3. Determine the most likely score (1-5) or if they indicated skipping.
                        4. State the score you are proposing clearly (e.g., 'I'll mark that as a [score]').
                        5. Explicitly ask for confirmation: 'Does that feel right?'. Do NOT ask the next BFI question yet.
                        """
                        if history_for_generation: history_for_generation[-1]['content'] = f"{history_for_generation[-1]['content']}\n\n[System instruction for interpreting DIRECT answer to QID {target_qid}: {instruction}]"
                        ai_response_text = conversation_handler.generate_response(history_for_generation, **config.get_generation_params())
                        needs_generation = False

                        proposed_answer_details = response_parser.extract_confirmed_answer(ai_response_text)
                        if proposed_answer_details:
                            logger.info(f"AI proposed answer for QID {target_qid}: {proposed_answer_details}")
                            session_state['pending_answer'] = {'value': proposed_answer_details['value'], 'type': proposed_answer_details['type']}
                            session_state['state'] = session_manager.STATES['AWAITING_CONFIRMATION']
                            session_state['last_question_id'] = target_qid
                            session_state['current_question_id'] = None
                        else: # AI failed to propose score even for direct answer
                            logger.warning(f"AI failed to propose score for direct answer '{user_message}' to QID {target_qid}.")
                            # Ask user directly for score
                            ai_response_text = f"Okay, thanks. To confirm, how would you rate '{question_text_full}' (1-5 or skip)?"
                            # Remain IN_PROGRESS, keep current_qid
                            session_state['pending_answer'] = None
                        # --- End score proposal logic ---
                    else:
                        # *** User message is conversational or a question ***
                        logger.debug(f"User message '{user_message}' looks conversational. Generating conversational response.")
                        # Instruct AI to respond conversationally and then RE-ASK the question.
                        instruction = f"""
                        The user didn't provide a direct score rating in their last message ('{user_message}').
                        Instead, respond conversationally to their message. Address their question or comment appropriately (e.g., provide a brief example or explanation if asked and possible, or acknowledge their thought).
                        After your conversational response, you MUST re-ask the original question clearly. Phrase it like:
                        "So, thinking again about the statement: '{question_text_full}', how much do you agree with it?"
                        or similar phrasing that brings the focus back to the BFI item.
                        Do NOT propose a score yet.
                        """
                        if history_for_generation: history_for_generation[-1]['content'] = f"{history_for_generation[-1]['content']}\n\n[System instruction for conversational response and re-asking QID {target_qid}: {instruction}]"

                        ai_response_text = conversation_handler.generate_response(
                            history_for_generation,
                            **config.get_generation_params()
                        )
                        needs_generation = False
                        # Keep state IN_PROGRESS and current_qid the same, clear any stale pending answer
                        session_state['state'] = session_manager.STATES['IN_PROGRESS']
                        session_state['current_question_id'] = target_qid
                        session_state['pending_answer'] = None
                        logger.debug(f"Generated conversational response for QID {target_qid}. State remains IN_PROGRESS.")
                        # *** End conversational handling ***

            # --- Handle other states (COMPLETED, NOT_STARTED, ERROR) ---
            elif current_internal_state == session_manager.STATES['COMPLETED']:
                logger.info(f"Chat received for completed session {sid}.")
                ai_response_text = "The interview has already concluded. Thank you!"
                needs_generation = False

            elif current_internal_state == session_manager.STATES['NOT_STARTED']:
                logger.warning(f"Chat received but interview not started for session {sid}.")
                ai_response_text = "The interview hasn't started yet. Please click 'Start Interview'."
                needs_generation = False

            else: # Error state or unknown
                logger.error(f"Chat received in unexpected state: {current_internal_state} for session {sid}.")
                ai_response_text = "Sorry, the interview encountered an unexpected state."
                session_state['state'] = session_manager.STATES['ERROR']
                needs_generation = False

            # --- Add AI Response to Actual History ---
            session_state = session_manager.add_message_to_history(session_state, 'model', ai_response_text)

            # --- Final Save and Return ---
            # (Save and return logic remains the same as the previous version)
            session['session_state'] = session_state
            logger.debug(f"Session state saved after processing chat for session {sid}")
            stats = session_manager.get_interview_stats(session_state)
            final_state = session_state.get('state')
            highlight_qid = None
            if final_state == session_manager.STATES['AWAITING_CONFIRMATION']:
                highlight_qid = session_state.get('last_question_id')
            elif final_state == session_manager.STATES['IN_PROGRESS']:
                highlight_qid = session_state.get('current_question_id')
            is_completed = final_state == session_manager.STATES['COMPLETED']
            logger.info(f"--- End processing chat for {sid} ---")
            logger.info(f"Returning State: {final_state}, Highlight QID: {highlight_qid}, Form Update: {form_update}")
            return jsonify({
                'success': final_state != session_manager.STATES['ERROR'],
                'ai_message': ai_response_text,
                'interview_state': final_state,
                'form_update': form_update,
                'current_question_id': highlight_qid,
                'is_completed': is_completed,
                'progress': stats.get('progress_percentage', 0)
            })

        except Exception as e:
            logger.error(f"Critical error processing chat: {e}", exc_info=True)
            # (Error handling as before)
            try:
                if 'session_state' in session and isinstance(session['session_state'], dict):
                    session['session_state']['state'] = session_manager.STATES['ERROR']
                    session['session_state']['last_updated'] = datetime.now().isoformat()
                    session.modified = True
                else:
                    session['session_state'] = {'state': session_manager.STATES['ERROR']}
            except Exception as save_e:
                logger.error(f"Failed to save error state to session: {save_e}")
            return jsonify({'success': False,'error': 'Critical error processing chat request'}), 500


    # --- Other Routes ---
    @app.route('/api/results', methods=['GET'])
    def get_results():
         # ... (Implementation as previously corrected) ...
        if not bfi_scorer: return jsonify({'success': False, 'error': 'Results module not ready'}), 500
        if 'session_state' not in session or not isinstance(session.get('session_state'), dict): return jsonify({'error': 'No active session.'}), 400
        session_state = session['session_state']; sid = getattr(session, 'sid', 'N/A')
        if session_state.get('state') != session_manager.STATES['COMPLETED']: return jsonify({'error': 'Interview not completed.'}), 400
        answers = session_state.get('answered_questions', {}); logger.info(f"Generating results for session {sid} ({len(answers)} answers).")
        try:
            report = bfi_scorer.generate_comprehensive_report(answers); report['session_id'] = sid; report['timestamp'] = datetime.now().isoformat()
            logger.info(f"Results generated successfully for session {sid}"); return jsonify({'success': True, 'report': report})
        except Exception as e: logger.error(f"Error generating results: {e}", exc_info=True); return jsonify({'success': False, 'error': 'Error generating results report'}), 500


    @app.route('/clear_session')
    def clear_session_route():
         # ... (Implementation as previously corrected) ...
        sid = getattr(session, 'sid', 'N/A'); session.clear(); logger.info(f"Session {sid} cleared by request.")
        return "Session cleared. <a href='/'>Return to home</a>"


    @app.route('/model_info')
    def model_info():
         # ... (Implementation as previously corrected) ...
         if not model_manager: return jsonify({'success': False,'error': 'Model manager not initialized'}), 500
         try: info = model_manager.get_model_info(); return jsonify({'success': True, 'model_info': info})
         except Exception as e: logger.error(f"Error getting model info: {e}", exc_info=True); return jsonify({'success': False, 'error': 'Could not retrieve model info'}), 500


    return app

# --- Main Execution Block ---
if __name__ == '__main__':
    try:
        app = create_app()
        logger.info(f"Starting Flask server on {config.HOST}:{config.PORT} (Debug Mode: {config.DEBUG_MODE})")
        if config.DEBUG_MODE: app.run(debug=True, host=config.HOST, port=config.PORT, use_reloader=False) # Keep reloader false
        else: from waitress import serve; logger.info("Running in production mode with Waitress."); serve(app, host=config.HOST, port=config.PORT)
    except RuntimeError as e: logger.critical(f"App init error: {e}")
    except Exception as e: logger.critical(f"Unexpected startup error: {e}", exc_info=True)