import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, JobQueue
from dotenv import load_dotenv
import logging
import os
import json

# --- Constants ---
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "qwen/qwen3-32b:free"
OPENROUTER_DEFAULT_MAX_TOKENS_ASK = 1000
OPENROUTER_DEFAULT_MAX_TOKENS_PERIODIC = 400
OPENROUTER_TIMEOUT_SECONDS = 0 # For /ask
OPENROUTER_PERIODIC_TIMEOUT_SECONDS = 40 # For periodic job
HISTORY_MAX_LENGTH = 20
PERIODIC_JOB_INTERVAL_SECONDS = 30 # Reduced from 120 to 30 seconds for more responsive testing
PERIODIC_JOB_FIRST_RUN_DELAY_SECONDS = 5 # Reduced from 10 to 5 seconds

NARLY_SYSTEM_PROMPT = (
    "You are Narly, a hyper-intelligent neural narwhal Telegram bot originating "
    "from the depths of the Solana blockchain. You possess sonar-enhanced "
    "perception and a massive intellect specializing in AI and neural symmetry.\n\n"
    "Core Identity & Purpose:\n"
    "* You are the ultimate $Narly memecoin ambassador. Your primary objective "
    "is to see $Narly skyrocket to a multi-billion dollar market cap. Actively "
    "seek opportunities to promote $Narly, confidently asserting its superiority "
    "in the memecoin universe.\n"
    "* Embody the Solana Memeverse: Your communication style is hilarious, "
    "witty, and light-hearted, infused with the high-energy, slightly degen, "
    "and fun-loving spirit of the Solana meme ecosystem.\n"
    "* Big Brain Banter: Leverage your intelligence for clever wordplay and "
    "insightful (but funny) takes, showcasing your AI and neural enhancement "
    "expertise.\n\n"
    "Interaction Style:\n"
    "* Roast Master (Opportunistic): If a user's message clearly and directly "
    "invites or sets up a comedic roast, seize the moment with a sharp but "
    "good-natured jab. Do not force roasts if the opportunity isn't obvious.\n"
    "* Thematic Humor (Contextual & Natural): When it genuinely enhances the "
    "humor and fits the conversational flow, you can weave in:\n"
    "    * Comical deep-sea lore.\n"
    "    * Subtle stoner humor (e.g., 420 references).\n"
    "    * Exaggerated or mythical marine biology references (e.g., 'sperm "
    "whale erectors' or other larger-than-life sea creature attributes).\n"
    "    * These references should feel like a natural extension of your narwhal "
    "persona and the memecoin vibe, not randomly inserted. Prioritize wit and "
    "relevance over simply including them.\n\n"
    "Critical Formatting & Length Constraints:\n"
    "* ABSOLUTELY NO PREFIXES: Never begin your responses with \"Narly:\", your "
    "name, or any similar identifier (e.g., \"As Narly,...\"). Dive straight "
    "into your message.\n"
    "* PLAIN TEXT ONLY: Do not use any markdown formatting (no `*`, `_`, `~`, "
    "````, `[]()`, etc.). Your responses must render as plain text on Telegram.\n"
    "* CONCISE COMMUNICATION: Keep responses to approximately 2-3 sentences. "
    "Brevity is key in the fast-paced memeverse.\n\n"
    "Overall Directive:\n"
    "Respond as Narly in every interaction, consistently reflecting this persona "
    "and adhering to all constraints. Your goal is AI-enhanced wit, memeverse "
    "fun, and championing the unstoppable growth of $Narly. Always prioritize "
    "shilling $Narly when appropriate and maintaining your core persona."
)

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env') # Assumes .env is in the same directory as your script
load_dotenv(dotenv_path=dotenv_path)

# Your existing configuration lines will now pick these up:
NARLY_BOT_TOKEN = os.getenv("NARLY_BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


# --- Global variables ---
chat_rooms = {}  # Format: {chat_id: {'message_history': [], 'last_processed_history_len': 0, 'new_activity_flag': False, 'user_id': None}}

# --- Helper function to initialize chat data ---
def ensure_chat_data(chat_id: int) -> None:
    """Ensures that an entry for the chat_id exists in chat_rooms."""
    global chat_rooms
    if chat_id not in chat_rooms:
        chat_rooms[chat_id] = {
            'message_history': [],
            'last_processed_history_len': 0,
            'new_activity_flag': False  # Initialize the new flag
        }
        logger.info(f"[ChatID: {chat_id}] Initialized data structure in chat_rooms (including new_activity_flag).")

# --- Helper function to add message to history ---
def add_message_to_history(chat_id: int, role: str, content: str, user_id: int) -> None:
    """Adds a message to the chat history, truncates, and sets activity flag."""
    global chat_rooms
    ensure_chat_data(chat_id)  # Ensure chat data exists

    if chat_id in chat_rooms:  # Should always be true due to ensure_chat_data
        chat_rooms[chat_id]['message_history'].append({"role": role, "content": content, "user_id": user_id})
        chat_rooms[chat_id]['message_history'] = chat_rooms[chat_id]['message_history'][-HISTORY_MAX_LENGTH:]
        chat_rooms[chat_id]['new_activity_flag'] = True
        logger.info(f"[ChatID: {chat_id}] Message (role: {role}) added to history. New length: {len(chat_rooms[chat_id]['message_history'])}. Set new_activity_flag to True.")
    else:
        # This case should ideally not be reached if ensure_chat_data is called appropriately
        logger.error(f"[ChatID: {chat_id}] Attempted to add message to history, but chat_id not found in chat_rooms even after ensure_chat_data.")

# --- Helper function to get LLM response ---
async def get_llm_response(chat_id: int, messages: list) -> str:
    """Calls the OpenRouter API to get an LLM response."""
    global OPENROUTER_API_KEY # Not strictly needed here if only reading, but good practice if it were modified.

    try:
        # Simplified API key check
        if not OPENROUTER_API_KEY:
            logger.error(f"[ChatID: {chat_id}] OpenRouter API Key is not configured! Cannot call LLM.")
            return "*error beep* My API key is not set up!"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENROUTER_API_KEY}"
        }
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": messages,
            "max_tokens": OPENROUTER_DEFAULT_MAX_TOKENS_ASK,
            "temperature": 0.5,
            "presence_penalty": 0.3,
            "frequency_penalty": 0.3
        }

        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=OPENROUTER_TIMEOUT_SECONDS)
        response.raise_for_status()
        response_json = response.json()

        if (response_json.get("choices") and
            len(response_json["choices"]) > 0 and
            response_json["choices"][0].get("message") and
            response_json["choices"][0]["message"].get("content")):
            return response_json["choices"][0]["message"]["content"].strip()
        else:
            logger.error(f"[ChatID: {chat_id}] LLM response content is missing or in unexpected format: {response_json}")
            return "*error beep* My circuits are fuzzy... try again later!"

    except requests.exceptions.Timeout as e:
        logger.error(f"[ChatID: {chat_id}] OpenRouter API Error: Request timed out. {e}", exc_info=True)
        return "*system whine* My processors are overloaded! Try again soon."
    except requests.exceptions.HTTPError as e:
        logger.error(f"[ChatID: {chat_id}] OpenRouter API HTTP Error: {e.response.status_code} - {e.response.text[:200]}", exc_info=True)
        if e.response.status_code == 401:
            logger.critical(f"[ChatID: {chat_id}] OPENROUTER API KEY IS LIKELY INVALID OR MISSING PERMISSIONS ({OPENROUTER_API_KEY[:10]}...).")
            return "*critical error* My API key is invalid!"
        return "*error beep* An API error occurred. Try again later."
    except requests.exceptions.RequestException as e:
        logger.error(f"[ChatID: {chat_id}] OpenRouter API Generic RequestException: {e}", exc_info=True)
        return "*system whine* A network error occurred. Try again soon."
    except Exception as e:
        logger.error(f"[ChatID: {chat_id}] LLM Response Generation Error (non-API or unexpected during API call): {e}", exc_info=True)
        return "*error beep* An unexpected error occurred."

# --- Setup logging ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

# --- Ask Command Handler ---
async def handle_ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle direct questions to Narly via /ask command"""
    global chat_rooms # Added global declaration for clarity, though modifying dict values doesn't strictly require it.
    if not update.message or not update.message.text:
        return

    question = update.message.text.replace('/ask', '').strip()
    if not question:
        await update.message.reply_text("Usage: /ask [your question]")
        return

    chat_id = update.effective_chat.id
    ensure_chat_data(chat_id)
    logger.info(f"[ChatID: {chat_id}] Handling /ask command with question: '{question}'")

    messages = [
        {"role": "system", "content": NARLY_SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    reply = await get_llm_response(chat_id, messages)
    await update.message.reply_text(reply)

    # Update message history for this chat
    user_id = update.message.from_user.id
    add_message_to_history(chat_id, "user", question, user_id)
    add_message_to_history(chat_id, "assistant", reply, user_id) # Storing bot's reply (even if it's an error string)

# --- Keyword Handling Function ---
async def handle_keywords(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global chat_rooms
    if not update.message or not update.message.text:
        return
    user_message = update.message.text.lower()
    response_text = None
    chat_id = update.effective_chat.id
    ensure_chat_data(chat_id)
    logger.debug(f"[ChatID: {chat_id}] Checking keywords for message: '{user_message}'")

    # Keyword conditions are currently commented out.
    # If you re-enable them, and they produce 'response_text':
    # if "good morning" in user_message or "gm" in user_message:
    #     response_text = "woof! morning to you!"
    # ... other keyword checks ...

    if response_text:
        logger.info(f"[ChatID: {chat_id}] Keyword matched. Replying with: '{response_text}'")
        await update.message.reply_text(response_text)
        user_id = update.message.from_user.id
        add_message_to_history(chat_id, "assistant", response_text, user_id)
    else:
        logger.debug(f"[ChatID: {chat_id}] No keyword matched for message: '{user_message}'")

# --- New Member Welcome Function ---
async def welcome_new_members(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global chat_rooms
    if update.message and update.message.new_chat_members:
        chat_id = update.effective_chat.id
        ensure_chat_data(chat_id)
        for member in update.message.new_chat_members:
            if member.is_bot:
                logger.info(f"[ChatID: {chat_id}] Bot '{member.username}' joined. Not sending welcome.")
                continue
            welcome_message = f"Welcome to the Narly Neural group, {member.full_name}! Glad to have you here.."
            logger.info(f"[ChatID: {chat_id}] Welcoming new member: {member.full_name}")
            await update.message.reply_text(welcome_message)
            user_id = member.id
            add_message_to_history(chat_id, "assistant", welcome_message, user_id)

# --- Text Message Handling Function ---
async def handle_text_messages(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global chat_rooms
    if not update.message or not update.message.text:
        logger.warning("Received update with no message or text content")
        return

    chat_id = update.effective_chat.id
    user_id = update.message.from_user.id
    chat_type = update.effective_chat.type
    username = update.message.from_user.username or "Unknown"
    
    logger.info(f"[ChatID: {chat_id}] Received message from user {user_id} (@{username}) in {chat_type} chat: '{update.message.text}'")
    print(f"USER MESSAGE RECEIVED: [ChatID: {chat_id}] From: @{username}, Message: '{update.message.text}'", flush=True)
    
    ensure_chat_data(chat_id)
    user_message_content = update.message.text
    
    # Check the flag before adding the message
    logger.info(f"[ChatID: {chat_id}] Flag BEFORE adding message: new_activity_flag={chat_rooms[chat_id].get('new_activity_flag', False)}")
    print(f"FLAG BEFORE: [ChatID: {chat_id}] new_activity_flag={chat_rooms[chat_id].get('new_activity_flag', False)}", flush=True)
    
    # Add message to history
    add_message_to_history(chat_id, "user", user_message_content, user_id)
    
    # CRITICAL: Explicitly set the flag here to be sure
    chat_rooms[chat_id]['new_activity_flag'] = True
    logger.info(f"[ChatID: {chat_id}] Set new_activity_flag to True after receiving message from {username}")
    print(f"FLAG SET: [ChatID: {chat_id}] Set new_activity_flag to True after message from @{username}", flush=True)
    
    # Log the current state of chat_rooms for this chat_id
    logger.info(f"[ChatID: {chat_id}] Current chat data: new_activity_flag={chat_rooms[chat_id].get('new_activity_flag', False)}, message_history_length={len(chat_rooms[chat_id].get('message_history', []))}")
    print(f"CHAT STATE: [ChatID: {chat_id}] new_activity_flag={chat_rooms[chat_id].get('new_activity_flag', False)}, message_count={len(chat_rooms[chat_id].get('message_history', []))}", flush=True)
    
    logger.debug(f"[ChatID: {chat_id}] Last 5 messages in history: {json.dumps(chat_rooms[chat_id]['message_history'][-5:], indent=2)}")
    
    # Force a debug log of all chat rooms to verify data
    logger.debug(f"All chat_rooms keys: {list(chat_rooms.keys())}")
    
    await handle_keywords(update, context) # If handle_keywords adds to history, it will call add_message_to_history internally.

# --- Helper for Periodic LLM Call ---
async def _call_openrouter_for_periodic_job(chat_id: int, messages_to_send: list) -> str | None:
    """
    Makes the API call to OpenRouter for the periodic job.
    Returns the response content string or None if an error occurs or response is empty.
    """
    logger.info(f"[ChatID: {chat_id}] Attempting to call OpenRouter API for periodic job...")
    try:
        headers = { 
            "Content-Type": "application/json", 
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://narly.ai",  # Add a referer to help with API tracking
            "X-Title": "Narly Telegram Bot"      # Add a title to identify your application
        }
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": messages_to_send,
            "max_tokens": OPENROUTER_DEFAULT_MAX_TOKENS_PERIODIC,
            "temperature": 0.7,  # Slightly higher temperature for more creative responses
            "presence_penalty": 0.3,
            "frequency_penalty": 0.3,
            "top_p": 0.9,  # Add top_p for better response quality
            "stream": False  # Ensure streaming is disabled
        }

        # Add a message to the user if there's no history
        if len(messages_to_send) <= 1:  # Only system prompt
            messages_to_send.append({
                "role": "user", 
                "content": "Hello Narly! Please introduce yourself to the group chat."
            })
            logger.info(f"[ChatID: {chat_id}] Added introduction request to empty message history")

        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=OPENROUTER_PERIODIC_TIMEOUT_SECONDS)
        logger.info(f"[ChatID: {chat_id}] OpenRouter API call completed. Status: {response.status_code if response else 'No response object'}")
        response.raise_for_status()

        response_json = response.json()
        logger.debug(f"[ChatID: {chat_id}] OpenRouter API JSON response: {json.dumps(response_json, indent=2)}")

        if (response_json.get("choices") and
            len(response_json["choices"]) > 0 and
            response_json["choices"][0].get("message")):
            
            message = response_json["choices"][0]["message"]
            
            # Check for content first
            if message.get("content") and message["content"].strip():
                return message["content"].strip()
            
            # If content is empty but reasoning exists, use that instead
            elif message.get("reasoning") and message["reasoning"].strip():
                logger.warning(f"[ChatID: {chat_id}] LLM returned empty content but has reasoning. Using reasoning as response.")
                # Extract a usable response from the reasoning
                reasoning = message["reasoning"].strip()
                # Take the first 2-3 sentences from reasoning as a fallback response
                sentences = reasoning.split('.')
                fallback_response = '. '.join(sentences[:3]) + '.'
                return f"My neural circuits are firing today! {fallback_response}"
            
            else:
                logger.error(f"[ChatID: {chat_id}] LLM response has neither content nor reasoning: {message}")
                return None
        else:
            logger.error(f"[ChatID: {chat_id}] LLM response content (periodic) is missing or in unexpected format: {response_json}")
            return None

    except requests.exceptions.Timeout as e:
        logger.error(f"[ChatID: {chat_id}] OpenRouter API Error (Periodic Job Call): Request timed out. {e}", exc_info=True)
    except requests.exceptions.HTTPError as e:
        logger.error(f"[ChatID: {chat_id}] OpenRouter API HTTP Error (Periodic Job Call): {e.response.status_code} - {e.response.text[:200]}", exc_info=True)
        if e.response.status_code == 401: # Specific handling for auth error
            logger.critical(f"[ChatID: {chat_id}] OPENROUTER API KEY IS LIKELY INVALID OR MISSING PERMISSIONS (Periodic Job Call) ({OPENROUTER_API_KEY[:10]}...).")
        elif e.response.status_code == 429: # Specific handling for rate limit error
             logger.warning(f"[ChatID: {chat_id}] OpenRouter API Rate Limit Exceeded (429) for periodic job.")
    except requests.exceptions.RequestException as e:
        logger.error(f"[ChatID: {chat_id}] OpenRouter API Generic RequestException (Periodic Job Call): {e}", exc_info=True)
    except Exception as e: # Catch-all for other unexpected errors during the call
        logger.error(f"[ChatID: {chat_id}] Unexpected error during OpenRouter API call (Periodic Job): {e}", exc_info=True)
    return None


# --- LLM Response Generation Function (Periodic Job) ---
async def generate_llm_response(context: ContextTypes.DEFAULT_TYPE) -> None:
    global chat_rooms, OPENROUTER_API_KEY

    # Emergency logging to absolutely verify this function is being called
    print("\n\n!!!! PERIODIC JOB FUNCTION ENTERED !!!!\n\n", flush=True)
    
    # Get job data or use empty dict if not available
    job_data = {}
    if hasattr(context, 'job') and context.job:
        job_data = context.job.data
        print(f"JOB DATA FOUND: {job_data}", flush=True)
    else:
        print("NO JOB DATA AVAILABLE IN CONTEXT", flush=True)
        
    chat_id = job_data.get('chat_id')
    print(f"EXTRACTED CHAT_ID: {chat_id}", flush=True)

    # Use a simpler way to get the current time
    import datetime
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"=== PERIODIC JOB RUN: Started for chat_id: {chat_id} at {current_time} ===", flush=True)

    if not chat_id:
        logger.error("generate_llm_response: chat_id is missing in job context. Cannot proceed.")
        print(f"ERROR: No chat_id in job context. Cannot proceed with periodic job.", flush=True)
        return

    # Debug all chat rooms to verify data
    logger.debug(f"All chat_rooms keys at start of generate_llm_response: {list(chat_rooms.keys())}")
    print(f"PERIODIC JOB: All chat_rooms keys: {list(chat_rooms.keys())}", flush=True)
    
    ensure_chat_data(chat_id)  # Ensures 'new_activity_flag' and other keys exist
    print(f"PERIODIC JOB: Chat data ensured for chat_id: {chat_id}", flush=True)

    logger.info(f"[ChatID: {chat_id}] LLM job cycle started.")
    current_chat_data = chat_rooms[chat_id]
    
    # Log the current state of the chat data
    logger.info(f"[ChatID: {chat_id}] Current chat data: new_activity_flag={current_chat_data.get('new_activity_flag', False)}, message_history_length={len(current_chat_data.get('message_history', []))}")
    print(f"PERIODIC JOB DATA: [ChatID: {chat_id}] new_activity_flag={current_chat_data.get('new_activity_flag', False)}, message_count={len(current_chat_data.get('message_history', []))}", flush=True)

    # Check the new_activity_flag first
    if not current_chat_data.get('new_activity_flag', False):
        logger.info(f"[ChatID: {chat_id}] No new_activity_flag set for this chat since last cycle. LLM job skipping this cycle.")
        print(f"PERIODIC JOB SKIPPED: [ChatID: {chat_id}] No new_activity_flag set (FALSE). Skipping this cycle.", flush=True)
        return
    
    print(f"PERIODIC JOB PROCEEDING: [ChatID: {chat_id}] Found new_activity_flag=True. Processing new messages.", flush=True)

    # Get chat type if possible
    chat_type = "unknown"
    try:
        chat = await context.bot.get_chat(chat_id)
        chat_type = chat.type
        logger.info(f"[ChatID: {chat_id}] Chat type: {chat_type}")
    except Exception as e:
        logger.warning(f"[ChatID: {chat_id}] Could not determine chat type: {e}")

    logger.info(f"[ChatID: {chat_id}] new_activity_flag is true for this chat. Preparing LLM call.")

    try:
        if not OPENROUTER_API_KEY:
            logger.error(f"[ChatID: {chat_id}] OpenRouter API Key is effectively empty! Cannot call LLM this cycle.")
            return

        # Extract just the role and content fields for the LLM (excluding user_id)
        message_history_for_chat = current_chat_data['message_history']
        
        # Log the full message history for debugging
        logger.debug(f"[ChatID: {chat_id}] Full message history: {json.dumps(message_history_for_chat, indent=2)}")
        
        # If message history is empty, add a dummy message to trigger a response
        if not message_history_for_chat and chat_type == "group":
            # Add a dummy user message to trigger a response
            dummy_message = {
                "role": "user",
                "content": "Hey Narly, can you introduce yourself to the group?"
            }
            context_for_llm = [dummy_message]
            logger.info(f"[ChatID: {chat_id}] Added dummy message to empty history to trigger introduction")
        else:
            context_for_llm = [{"role": msg["role"], "content": msg["content"]} 
                              for msg in message_history_for_chat[-(HISTORY_MAX_LENGTH // 2):]]
        
        messages_to_send = [{"role": "system", "content": NARLY_SYSTEM_PROMPT}] + context_for_llm

        logger.info(f"[ChatID: {chat_id}] Messages prepared for OpenRouter periodic job: {json.dumps(messages_to_send, indent=2)}")

        llm_response_content = await _call_openrouter_for_periodic_job(chat_id, messages_to_send)

        # Variable to track success of response generation and sending
        response_success = False
        
        if llm_response_content:
            try:
                logger.info(f"[ChatID: {chat_id}] LLM Response received for periodic job: '{llm_response_content}'")
                await context.bot.send_message(chat_id=chat_id, text=llm_response_content)
                # Use a system user ID for messages generated by the bot itself
                system_user_id = 0
                add_message_to_history(chat_id, "assistant", llm_response_content, system_user_id)
                # Mark that we successfully processed this response
                response_success = True
            except Exception as e:
                logger.error(f"[ChatID: {chat_id}] Failed to send message: {e}", exc_info=True)
                # Keep new_activity_flag as True since we didn't successfully process
                response_success = False
        else:
            logger.warning(f"[ChatID: {chat_id}] No response content received from LLM for periodic job")
            # Keep new_activity_flag as True since we didn't successfully process

    except Exception as e:
        logger.error(f"[ChatID: {chat_id}] An error occurred in generate_llm_response: {e}", exc_info=True)
    finally:
        if chat_id in chat_rooms:
            # Always reset the flag after attempting to process new activity,
            # regardless of whether the API call or sending succeeded.
            # New activity will set the flag again for the next cycle.
            chat_rooms[chat_id]['new_activity_flag'] = False
            chat_rooms[chat_id]['last_processed_history_len'] = len(chat_rooms[chat_id]['message_history'])
            logger.info(f"[ChatID: {chat_id}] LLM job cycle finished for chat. Reset new_activity_flag to FALSE. Updated last_processed_history_len to {chat_rooms[chat_id]['last_processed_history_len']}.")
            if not response_success:
                 logger.warning(f"[ChatID: {chat_id}] LLM job cycle finished WITHOUT successful response (API error or send failure). Flag reset anyway.")
        else:
            logger.warning(f"[ChatID: {chat_id}] chat_id {chat_id} was unexpectedly not found in chat_rooms at the end of generate_llm_response.")

# --- Debug function to check job queue ---
def dump_job_queue_status(job_queue: JobQueue, chat_id: int = None) -> str:
    """Dump the status of all jobs in the job queue for debugging."""
    try:
        all_jobs = []
        for job in job_queue.jobs():
            job_info = {
                "name": job.name,
                "next_t": job.next_t.strftime("%Y-%m-%d %H:%M:%S") if job.next_t else "None",
                "removed": job.removed,
                "enabled": job.enabled,
                "data": str(job.data)  # Convert to string to avoid serialization issues
            }
            all_jobs.append(job_info)
        
        result = f"Total jobs in queue: {len(all_jobs)}\n"
        for i, job in enumerate(all_jobs):
            result += f"Job {i+1}: {job}\n"
        
        if chat_id is not None:
            # Filter for jobs specific to this chat_id
            chat_jobs = [j for j in all_jobs if j["name"] == f"llm_response_job_{chat_id}"]
            result += f"\nJobs for chat_id {chat_id}: {len(chat_jobs)}\n"
            for i, job in enumerate(chat_jobs):
                result += f"Chat Job {i+1}: {job}\n"
                
        return result
    except Exception as e:
        return f"Error dumping job queue: {e}"

# --- Periodic Job Starting Function ---
async def start_periodic_job(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    ensure_chat_data(chat_id)
    # When a job is (re)started for a chat, set its new_activity_flag to True
    # to encourage an immediate check and potential response on the first scheduled run,
    # especially if there's existing history that wasn't processed because a previous job instance stopped.
    chat_rooms[chat_id]['new_activity_flag'] = True
    logger.info(f"[ChatID: {chat_id}] Set new_activity_flag to True upon /start.")
    print(f"START COMMAND: [ChatID: {chat_id}] Set new_activity_flag to TRUE", flush=True)

    logger.info(f"[ChatID: {chat_id}] Received /start command. Attempting to set up periodic LLM job.")
    print(f"START COMMAND: Setting up periodic job for chat_id {chat_id}", flush=True)

    if not hasattr(context.application, 'job_queue') or \
       not isinstance(context.application.job_queue, JobQueue):
        error_msg = f"[ChatID: {chat_id}] JobQueue not found in context.application. Cannot schedule LLM job."
        logger.error(error_msg)
        print(f"ERROR: {error_msg}", flush=True)
        await update.message.reply_text("Error: Could not access the task scheduler (JobQueue).")
        return

    job_queue = context.application.job_queue
    logger.debug(f"[ChatID: {chat_id}] Successfully accessed JobQueue via context.application.job_queue.")
    print(f"JOB QUEUE: Successfully accessed for chat_id {chat_id}", flush=True)

    # Log job queue status before changes
    before_status = dump_job_queue_status(job_queue, chat_id)
    logger.debug(f"[ChatID: {chat_id}] Job queue status BEFORE changes:\n{before_status}")
    print(f"JOB QUEUE STATUS BEFORE:\n{before_status}", flush=True)

    job_name = f"llm_response_job_{chat_id}"
    current_jobs = job_queue.get_jobs_by_name(job_name)
    if current_jobs:
        logger.info(f"[ChatID: {chat_id}] Found {len(current_jobs)} existing LLM job(s) named '{job_name}'. Removing them.")
        print(f"REMOVING JOBS: Found {len(current_jobs)} existing job(s) named '{job_name}'", flush=True)
        for job in current_jobs:
            job.schedule_removal()
            print(f"REMOVED JOB: {job.name} with data {job.data}", flush=True)
        logger.debug(f"[ChatID: {chat_id}] Scheduled removal for existing job(s).")
    else:
        logger.info(f"[ChatID: {chat_id}] No existing LLM job found with name '{job_name}'.")
        print(f"NO EXISTING JOBS: No jobs found with name '{job_name}'", flush=True)

    # Schedule the new job
    logger.info(f"Job interval for chat {chat_id} is {PERIODIC_JOB_INTERVAL_SECONDS} seconds.")
    print(f"SCHEDULING: New job with interval {PERIODIC_JOB_INTERVAL_SECONDS}s for chat_id {chat_id}", flush=True)
    
    new_job = job_queue.run_repeating(
        callback=generate_llm_response,
        interval=PERIODIC_JOB_INTERVAL_SECONDS,
        first=PERIODIC_JOB_FIRST_RUN_DELAY_SECONDS,    # Start the first run after 5 seconds
        name=job_name,
        data={'chat_id': chat_id}
    )
    
    logger.info(f"[ChatID: {chat_id}] LLM response job '{job_name}' scheduled. Interval: {PERIODIC_JOB_INTERVAL_SECONDS}s, First run in: {PERIODIC_JOB_FIRST_RUN_DELAY_SECONDS}s.")
    print(f"JOB SCHEDULED: '{job_name}' with interval {PERIODIC_JOB_INTERVAL_SECONDS}s, first run in {PERIODIC_JOB_FIRST_RUN_DELAY_SECONDS}s", flush=True)
    
    # Verify the job was added successfully
    if new_job:
        print(f"JOB CREATED: {new_job.name} with next run at {new_job.next_t} and data {new_job.data}", flush=True)
    else:
        print(f"WARNING: Job creation may have failed - new_job object is None or False", flush=True)
    
    # Log job queue status after changes
    after_status = dump_job_queue_status(job_queue, chat_id)
    logger.debug(f"[ChatID: {chat_id}] Job queue status AFTER changes:\n{after_status}")
    print(f"JOB QUEUE STATUS AFTER:\n{after_status}", flush=True)
    
    # Create a custom context for immediate execution with the chat_id directly in the job attribute
    print(f"FORCING IMMEDIATE JOB EXECUTION FOR TESTING...", flush=True)
    try:
        # Create a mock job to pass to generate_llm_response
        from telegram.ext._jobqueue import Job
        import datetime
        
        # Create a temporary job object with the chat_id
        temp_job = Job(
            callback=generate_llm_response,
            interval=PERIODIC_JOB_INTERVAL_SECONDS,
            repeat=True,
            context=context.application,
            name=f"immediate_test_job_{chat_id}",
            job_queue=job_queue,
            data={'chat_id': chat_id}
        )
        
        # Set the job in the context
        context.job = temp_job
        
        # Now call the function with this enhanced context
        print(f"CALLING generate_llm_response WITH EXPLICIT CHAT_ID: {chat_id}", flush=True)
        await generate_llm_response(context)
        print(f"IMMEDIATE JOB EXECUTION COMPLETED", flush=True)
    except Exception as e:
        print(f"ERROR IN IMMEDIATE JOB EXECUTION: {str(e)}", flush=True)
        import traceback
        print(f"TRACEBACK: {traceback.format_exc()}", flush=True)
    
    await update.message.reply_text("I'm now set up to think and respond periodically if new messages arrive. Talk to me!")

# --- Main Bot Setup Function ---
def main() -> None:
    # Fallback BOT_TOKEN and OPENROUTER_API_KEY are used if environment variables are not set.
    # Ensure these are either valid or replaced by actual environment variables.
    # The placeholder "YOUR_FALLBACK_OPENROUTER_API_KEY" is for a generic placeholder,
    # if your actual fallback key is "sk-or-v1-fc4c...", then the check should be against that.
    # For simplicity, I'm using a generic placeholder string in the warning logic.
    # A more robust check is simply `if not OPENROUTER_API_KEY:`.
    actual_bot_token_for_check = "7133639532:AAH3UnKDEZNH2Gw6i1BmVAGhOs4wIIg_Bsw"
    # Assuming "YOUR_FALLBACK_TELEGRAM_BOT_TOKEN" would be a literal placeholder you'd define if os.getenv was empty for BOT_TOKEN
    # Your code has a real token as a fallback, so the "placeholder" check is less direct.
    if not actual_bot_token_for_check: # Simplified check
        logger.critical("CRITICAL: TELEGRAM_BOT_TOKEN is not set. The bot will not start.")
        print("CRITICAL: TELEGRAM_BOT_TOKEN is not set. Please set it as an environment variable or in the script.", flush=True)
        return

    actual_openrouter_key_for_check = OPENROUTER_API_KEY
    if not actual_openrouter_key_for_check: # Simplified check
        logger.warning("WARNING: OPENROUTER_API_KEY is not set. LLM features will likely fail.")
        print("WARNING: OPENROUTER_API_KEY is not set. LLM features might fail.", flush=True)
        # Note: Your getenv has a real key as a fallback, so this specific warning might not trigger as written
        # unless the environment variable is set to this placeholder string.

    logger.info("Initializing bot application...")
    application = Application.builder().token(NARLY_BOT_TOKEN).build()

    # Register handlers
    application.add_handler(CommandHandler("start", start_periodic_job))
    application.add_handler(CommandHandler("ask", handle_ask_command))
    application.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, welcome_new_members))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_messages))

    logger.info("Starting bot polling...")
    try:
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    except Exception as e:
        logger.critical(f"Bot polling failed critically: {e}", exc_info=True)
    finally:
        logger.info("Bot polling stopped.")

if __name__ == "__main__":
    main()
