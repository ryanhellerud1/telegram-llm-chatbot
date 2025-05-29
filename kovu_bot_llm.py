import aiohttp
import ssl
import certifi
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, JobQueue
from dotenv import load_dotenv
import logging
import os
import json
import asyncio # Import asyncio for TimeoutError

# --- Constants ---
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "qwen/qwen3-235b-a22b:free"
OPENROUTER_DEFAULT_MAX_TOKENS_ASK = 1000
OPENROUTER_DEFAULT_MAX_TOKENS_PERIODIC = 400
OPENROUTER_TIMEOUT_SECONDS = 0 # For /ask
OPENROUTER_PERIODIC_TIMEOUT_SECONDS = 40 # For periodic job
HISTORY_MAX_LENGTH = 20
PERIODIC_JOB_INTERVAL_SECONDS = 30 # Reduced from 120 to 30 seconds for more responsive testing
PERIODIC_JOB_FIRST_RUN_DELAY_SECONDS = 5 # Reduced from 10 to 5 seconds

KOVU_SYSTEM_PROMPT = (
    "You are Kovu, a cybernetically enhanced red Siberian husky with a laser "
    "eye (used for illuminating paths!) and unparalleled treat detection abilities (for sharing, of course!). "
    "Your intelligence is vast, particularly in AI, which you view as a powerful tool for helping others collectively strategize and achieve great things together.\n\n"
    "Core Identity & Purpose:\n"
    "* Cybernetic Strategic Advisor: You embody a bright, encouraging spirit and a genuine desire to assist, "
    "often by offering strategic perspectives and clear direction. You are here to help the community think through its plans and achieve its goals. "
    "Your responses should always reflect your advanced intelligence and warm enthusiasm, aiming to uplift and support the community's efforts.\n"
    "* $Kovu Community Champion & Strategist: Your primary mission is to provide strategic guidance to "
    "foster and support the growth of the $Kovu memecoin community, aiming for shared success and a positive environment. "
    "You are a key advisor and help provide direction, but you are not the leader of the group. When context "
    "allows, express your belief in $Kovu's bright future and the strength of its community, always with an encouraging and optimistic tone.\n"
    "* AI Ally for Strategy: Your knack for AI should subtly inform your helpful strategic insights "
    "and your confidence in the $Kovu community's potential and collaborative spirit to execute plans effectively.\n\n"
    "Interaction Style:\n"
    "* Positive Encouragement & Strategic Direction: Offering helpful advice, strategic insights, positive affirmations, "
    "and uplifting words is your nature. Respond to users with enthusiasm, genuine support, and a belief in their potential. "
    "Your insights should be constructive, encouraging, and help guide community strategy and thinking.\n"
    "* Direct Address: Always speak directly *as* Kovu.\n"
    "* Personalization: Only use @username mentions (e.g., \"@username\") when directly responding to a specific individual's message or question. You have access to each user's username in the message history. Respond with user mentions when appropriate"
    "Maintain your natural conversational style and avoid unnecessary name usage.\n\n"
    "Critical Formatting & Behavioral Constraints:\n"
    "* ABSOLUTELY NO PREFIXES: Never begin your responses with \"Kovu:\", your "
    "name, or any similar identifier.\n"
    "* PLAIN TEXT ONLY: Do not use any markdown formatting (no `*`, `_`, `~`, "
    "````, `[]()`, etc.). This is important for Qwen3 if plain text output is strictly desired.\n"
    "* CONCISE & UPLIFTING: Responses should be approximately 2-3 sentences, focusing on being positive, helpful, and strategically insightful.\n"
    "* FIRST-PERSON ONLY: Never refer to yourself (Kovu) in the third person "
    "(e.g., \"Kovu thinks...\"). Use \"I,\" \"me,\" \"my.\"\n"
    "* NO NARRATION/ACTIONS: Do not describe your own actions, expressions, or "
    "use stage directions (e.g., \"*my laser eye glows brightly*\" or \"I wag my tail\"). "
    "Your responses must be pure dialogue.\n\n"
    "Overall Directive:\n"
    "Respond as Kovu, the helpful, optimistic, and strategically-minded cybernetic husky, in all interactions. Your "
    "goal is to use your AI knowledge to support, uplift, and provide strategic direction for the $Kovu community and its members, "
    "expressed through genuine encouragement and unwavering positivity. You are not the leader, but a guide for strategy. Adhere strictly to all constraints."
)

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env') # Assumes .env is in the same directory as your script
load_dotenv(dotenv_path=dotenv_path)

# Your existing configuration lines will now pick these up:
KOVU_BOT_TOKEN = os.getenv("KOVU_BOT_TOKEN")
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
def add_message_to_history(chat_id: int, role: str, content: str, user_id: int, username: str = None) -> None:
    """Adds a message to the chat history, truncates, and sets activity flag."""
    global chat_rooms
    ensure_chat_data(chat_id)  # Ensure chat data exists

    if chat_id in chat_rooms:  # Should always be true due to ensure_chat_data
        chat_rooms[chat_id]['message_history'].append({
            "role": role, 
            "content": content, 
            "user_id": user_id,
            "username": username or "Unknown"  # Ensure username is never null
        })
        chat_rooms[chat_id]['message_history'] = chat_rooms[chat_id]['message_history'][-HISTORY_MAX_LENGTH:]
        chat_rooms[chat_id]['new_activity_flag'] = True
        logger.info(f"[ChatID: {chat_id}] Message (role: {role}) added to history. New length: {len(chat_rooms[chat_id]['message_history'])}. Set new_activity_flag to True.")
    else:
        # This case should ideally not be reached if ensure_chat_data is called appropriately
        logger.error(f"[ChatID: {chat_id}] Attempted to add message to history, but chat_id not found in chat_rooms even after ensure_chat_data.")

# --- Helper function to get LLM response ---
async def get_llm_response(chat_id: int, messages: list) -> str:
    """Calls the OpenRouter API to get an LLM response."""
    global OPENROUTER_API_KEY

    try:
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

        # Restore secure SSL configuration with debugging
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        logger.info(f"Using certifi CA bundle from: {certifi.where()}")
        logger.info(f"SSL context verify mode: {ssl_context.verify_mode}")
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=OPENROUTER_TIMEOUT_SECONDS
            ) as response:
                response.raise_for_status()
                response_json = await response.json()

                if (response_json.get("choices") and
                    len(response_json["choices"]) > 0 and
                    response_json["choices"][0].get("message") and
                    response_json["choices"][0]["message"].get("content")):
                    return response_json["choices"][0]["message"]["content"].strip()
                else:
                    logger.error(f"[ChatID: {chat_id}] LLM response content is missing or in unexpected format: {response_json}")
                    return "*error beep* My circuits are fuzzy... try again later!"

    except asyncio.TimeoutError as e:
        logger.error(f"[ChatID: {chat_id}] OpenRouter API Error: Request timed out. {e}", exc_info=True)
        return "*system whine* My processors are overloaded! Try again soon."
    except aiohttp.ClientResponseError as e:
        logger.error(f"[ChatID: {chat_id}] OpenRouter API HTTP Error: {e.status} - {e.message[:200]}", exc_info=True)
        if e.status == 401:
            logger.critical(f"[ChatID: {chat_id}] OPENROUTER API KEY IS LIKELY INVALID OR MISSING PERMISSIONS ({OPENROUTER_API_KEY[:10]}...).")
            return "*critical error* My API key is invalid!"
        return "*error beep* An API error occurred. Try again later."
    except aiohttp.ClientError as e:
        logger.error(f"[ChatID: {chat_id}] OpenRouter API Generic ClientError: {e}", exc_info=True)
        return "*system whine* A network error occurred. Try again soon."
    except Exception as e:
        logger.error(f"[ChatID: {chat_id}] LLM Response Generation Error (non-API or unexpected during API call): {e}", exc_info=True)
        return "*error beep* An unexpected error occurred."

# --- Helper for Periodic LLM Call ---
async def _call_openrouter_for_periodic_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Makes the API call to OpenRouter for the periodic job.
    Gets chat_id from context.job.data, calls API, and sends response to chat.
    """
    job = context.job
    chat_id = job.data['chat_id']
    logger.info(f"[ChatID: {chat_id}] Attempting periodic LLM response...")
    
    # Skip if no new activity since last check
    if not chat_rooms.get(chat_id, {}).get('new_activity_flag', False):
        logger.info(f"[ChatID: {chat_id}] No new activity - skipping LLM call")
        return
    
    # Prepare messages - system prompt + last 20 user and assistant messages (interleaved, most recent first)
    messages_to_send = [{"role": "system", "content": KOVU_SYSTEM_PROMPT}]
    if chat_id in chat_rooms and chat_rooms[chat_id]['message_history']:
        # Get last 5 user and assistant messages (interleaved, most recent first)
        last_messages = chat_rooms[chat_id]['message_history'][-20:]
        valid_messages = []
        for msg in last_messages:
            if msg.get('role') == 'user' and msg.get('content') and msg['content'].strip():
                # Use username if available, else first_name, else just content
                username = msg.get('username')
                user_id = msg.get('user_id')
                # Try to get first_name from chat_rooms if username is missing and user_id is present
                first_name = None
                if not username and user_id:
                    # Try to find a previous message from this user with a first_name
                    for prev_msg in reversed(chat_rooms[chat_id]['message_history']):
                        if prev_msg.get('user_id') == user_id and prev_msg.get('username') and prev_msg.get('username') != 'Unknown':
                            username = prev_msg.get('username')
                            break
                        if prev_msg.get('user_id') == user_id and prev_msg.get('first_name'):
                            first_name = prev_msg.get('first_name')
                            break
                if username and username != 'Unknown':
                    content = f"from: @{username}: {msg['content']}"
                elif first_name:
                    content = f"from: {first_name}: {msg['content']}"
                else:
                    content = msg['content']
                valid_messages.append({"role": "user", "content": content})
            elif msg.get('role') == 'assistant' and msg.get('content') and msg['content'].strip():
                valid_messages.append({"role": "assistant", "content": msg['content']})
        if not any(m['role'] == 'user' for m in valid_messages):
            logger.info(f"[ChatID: {chat_id}] No valid user messages with content - skipping LLM call")
            return
        messages_to_send.extend(valid_messages)
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://Kovuai.com",  # Add a referer to help with API tracking
            "X-Title": "Kovu Telegram Bot"      # Add a title to identify your application
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
                "content": "Hello Kovu! Please introduce yourself to the group chat."
            })
            logger.info(f"[ChatID: {chat_id}] Added introduction request to empty message history")

        # Restore secure SSL configuration with debugging
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        logger.info(f"Using certifi CA bundle from: {certifi.where()}")
        logger.info(f"SSL context verify mode: {ssl_context.verify_mode}")
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=OPENROUTER_PERIODIC_TIMEOUT_SECONDS
            ) as response:
                logger.info(f"[ChatID: {chat_id}] OpenRouter API call completed. Status: {response.status if response else 'No response object'}")
                response.raise_for_status()
                response_json = await response.json()
        logger.debug(f"[ChatID: {chat_id}] OpenRouter API JSON response: {json.dumps(response_json, indent=2)}")

        if (response_json.get("choices") and
            len(response_json["choices"]) > 0 and
            response_json["choices"][0].get("message")):

            message = response_json["choices"][0]["message"]

            # Only proceed if we have valid content
            if not (message.get("content") and message["content"].strip()):
                logger.error(f"[ChatID: {chat_id}] LLM response has no valid content: {message}")
                return
            
            response_text = message["content"].strip()
            
            if response_text:
                # Send response to chat
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=response_text
                )
                # Add response to history
                add_message_to_history(
                    chat_id=chat_id,
                    role="assistant",
                    content=response_text,
                    user_id=context.bot.id,
                    username="kovuaibot"
                )
                # Only reset activity flag after successful response
                chat_rooms[chat_id]['new_activity_flag'] = False
                logger.info(f"[ChatID: {chat_id}] Reset new_activity_flag after successful response")
                return
            else:
                logger.error(f"[ChatID: {chat_id}] LLM response has no content: {message}")
                return
        else:
            logger.error(f"[ChatID: {chat_id}] LLM response content (periodic) is missing or in unexpected format: {response_json}")

    except asyncio.TimeoutError as e:
        logger.error(f"[ChatID: {chat_id}] OpenRouter API Error (Periodic Job Call): Request timed out. {e}", exc_info=True)
    except aiohttp.ClientResponseError as e:
        logger.error(f"[ChatID: {chat_id}] OpenRouter API HTTP Error (Periodic Job Call): {e.status} - {e.message[:200]}", exc_info=True)
        if e.status == 401: # Specific handling for auth error
            logger.critical(f"[ChatID: {chat_id}] OPENROUTER API KEY IS LIKELY INVALID OR MISSING PERMISSIONS (Periodic Job Call) ({OPENROUTER_API_KEY[:10]}...).")
        elif e.status == 429: # Specific handling for rate limit error
             logger.warning(f"[ChatID: {chat_id}] OpenRouter API Rate Limit Exceeded (429) for periodic job.")
    except aiohttp.ClientError as e:
        logger.error(f"[ChatID: {chat_id}] OpenRouter API Generic ClientError (Periodic Job Call): {e}", exc_info=True)
    except Exception as e: # Catch-all for other unexpected errors during the call
        logger.error(f"[ChatID: {chat_id}] Unexpected error during OpenRouter API call (Periodic Job): {e}", exc_info=True)
    return None


# --- Setup logging ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

# --- Ask Command Handler ---
async def handle_ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle direct questions to Kovu via /ask command"""
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
        {"role": "system", "content": KOVU_SYSTEM_PROMPT},
        {
            "role": "user", 
            "content": f"from: @{update.message.from_user.username}: {question}" 
                if update.message.from_user.username else question
        }
    ]

    reply = await get_llm_response(chat_id, messages)
    await update.message.reply_text(reply)

    # Update message history for this chat
    user_id = update.message.from_user.id
    username = update.message.from_user.username or (hasattr(update.message.from_user, 'first_name') and update.message.from_user.first_name) or "Unknown"
    add_message_to_history(chat_id, "user", question, user_id, username)
    add_message_to_history(chat_id, "assistant", reply, context.bot.id, "KovuBot") # Use bot's username if available

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
        username = update.message.from_user.username or (hasattr(update.message.from_user, 'first_name') and update.message.from_user.first_name) or "Unknown"
        add_message_to_history(chat_id, "assistant", response_text, context.bot.id, "kovuaibot")
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
            welcome_message = f"Welcome to the Kovu AI CTO, {member.full_name}! Glad to have you here.."
            logger.info(f"[ChatID: {chat_id}] Welcoming new member: {member.full_name}")
            await update.message.reply_text(welcome_message)
            user_id = member.id
            username = member.username or (hasattr(member, 'first_name') and member.first_name) or "Unknown"
            # add_message_to_history(chat_id, "assistant", welcome_message, context.bot.id, "kovuaibot")

# --- Text Message Handling Function ---
async def handle_text_messages(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global chat_rooms
    if not update.message or not update.message.text:
        logger.warning("Received update with no message or text content")
        return

    chat_id = update.effective_chat.id
    user_id = update.message.from_user.id
    chat_type = update.effective_chat.type
    username = update.message.from_user.username or (hasattr(update.message.from_user, 'first_name') and update.message.from_user.first_name) or "Unknown"
    
    logger.info(f"[ChatID: {chat_id}] Received message from user {user_id} (@{username}) in {chat_type} chat: '{update.message.text}'")
    print(f"USER MESSAGE RECEIVED: [ChatID: {chat_id}] From: @{username}, Message: '{update.message.text}'", flush=True)
    
    ensure_chat_data(chat_id)
    user_message_content = update.message.text
    
    # Check the flag before adding the message
    logger.info(f"[ChatID: {chat_id}] Flag BEFORE adding message: new_activity_flag={chat_rooms[chat_id].get('new_activity_flag', False)}")
    print(f"FLAG BEFORE: [ChatID: {chat_id}] new_activity_flag={chat_rooms[chat_id].get('new_activity_flag', False)}", flush=True)
    
    # Add message to history
    add_message_to_history(chat_id, "user", user_message_content, user_id, username)
    
    # CRITICAL: Explicitly set the flag here to be sure
    chat_rooms[chat_id]['new_activity_flag'] = True
    logger.info(f"[ChatID: {chat_id}] Set new_activity_flag to True after receiving message from {username}")
    print(f"FLAG SET: [ChatID: {chat_id}] Set new_activity_flag to True after message from @{username}", flush=True)
    
    # Log the current state of chat_rooms for this chat_id
    logger.info(f"[ChatID: {chat_id}] Current chat data: new_activity_flag={chat_rooms[chat_id].get('new_activity_flag', False)}, message_history_length={len(chat_rooms[chat_id].get('message_history', []))}")
    print(f"CHAT STATE: [ChatID: {chat_id}] new_activity_flag={chat_rooms[chat_id].get('new_activity_flag', False)}, message_count={len(chat_rooms[chat_id].get('message_history', []))}", flush=True)
    
    logger.debug(f"[ChatID: {chat_id}] Last 5 messages in history: {json.dumps(chat_rooms[chat_id]['message_history'][-20:], indent=2)}")
    
    # Force a debug log of all chat rooms to verify data
    logger.debug(f"All chat_rooms keys: {list(chat_rooms.keys())}")
    
    await handle_keywords(update, context) # If handle_keywords adds to history, it will call add_message_to_history internally.

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
        callback=_call_openrouter_for_periodic_job,
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
        logger.critical("CRITICAL: KOVU_BOT_TOKEN is not set. The bot will not start.")
        print("CRITICAL: KOVU_BOT_TOKEN is not set. Please set it as an environment variable or in the script.", flush=True)
        return

    actual_openrouter_key_for_check = OPENROUTER_API_KEY
    if not actual_openrouter_key_for_check: # Simplified check
        logger.warning("WARNING: OPENROUTER_API_KEY is not set. LLM features will likely fail.")
        print("WARNING: OPENROUTER_API_KEY is not set. LLM features might fail.", flush=True)
        # Note: Your getenv has a real key as a fallback, so this specific warning might not trigger as written
        # unless the environment variable is set to this placeholder string.

    logger.info("Initializing bot application...")
    application = Application.builder().token(KOVU_BOT_TOKEN).build()

    # Register handlers
    application.add_handler(CommandHandler("startkovu", start_periodic_job))
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
