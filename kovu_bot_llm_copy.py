import aiohttp
import ssl
import certifi
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, JobQueue
from dotenv import load_dotenv
import logging
import os
import json
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ChatData:
    """Data structure for individual chat information."""
    message_history: List[Dict[str, Any]] = field(default_factory=list)
    last_processed_history_len: int = 0
    new_activity_flag: bool = False
    
    def add_message(self, role: str, content: str, user_id: int, username: str = None) -> None:
        """Add a message to the chat history."""
        self.message_history.append({
            "role": role,
            "content": content,
            "user_id": user_id,
            "username": username or "Unknown",
            "timestamp": datetime.now().isoformat()
        })
        # Keep only the last HISTORY_MAX_LENGTH messages
        if len(self.message_history) > KovuBot.HISTORY_MAX_LENGTH:
            self.message_history = self.message_history[-KovuBot.HISTORY_MAX_LENGTH:]
        self.new_activity_flag = True


class KovuBot:
    """Main bot class for Kovu AI Telegram Bot."""
    
    # Constants
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
    OPENROUTER_MODEL = "qwen/qwen3-235b-a22b:free"
    OPENROUTER_DEFAULT_MAX_TOKENS_ASK = 1000
    OPENROUTER_DEFAULT_MAX_TOKENS_PERIODIC = 400
    OPENROUTER_TIMEOUT_SECONDS = 60  # Changed from 0 to reasonable timeout
    OPENROUTER_PERIODIC_TIMEOUT_SECONDS = 40
    HISTORY_MAX_LENGTH = 20
    PERIODIC_JOB_INTERVAL_SECONDS = 30
    PERIODIC_JOB_FIRST_RUN_DELAY_SECONDS = 5
    
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
    
    def __init__(self):
        """Initialize the bot with configuration and state."""
        self.logger = self._setup_logging()
        self._load_environment()
        self.chat_rooms: Dict[int, ChatData] = {}
        self.application: Optional[Application] = None
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.DEBUG
        )
        return logging.getLogger(__name__)
    
    def _load_environment(self) -> None:
        """Load environment variables."""
        dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
        load_dotenv(dotenv_path=dotenv_path)
        
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        
        if not self.bot_token:
            self.logger.critical("TELEGRAM_BOT_TOKEN is not set!")
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")
            
        if not self.openrouter_api_key:
            self.logger.warning("OPENROUTER_API_KEY is not set. LLM features will fail.")
    
    def _ensure_chat_data(self, chat_id: int) -> None:
        """Ensure chat data exists for the given chat_id."""
        if chat_id not in self.chat_rooms:
            self.chat_rooms[chat_id] = ChatData()
            self.logger.info(f"[ChatID: {chat_id}] Initialized chat data structure.")
    
    def _add_message_to_history(self, chat_id: int, role: str, content: str, 
                               user_id: int, username: str = None) -> None:
        """Add a message to chat history."""
        self._ensure_chat_data(chat_id)
        self.chat_rooms[chat_id].add_message(role, content, user_id, username)
        self.logger.info(f"[ChatID: {chat_id}] Message (role: {role}) added to history. "
                        f"New length: {len(self.chat_rooms[chat_id].message_history)}")
    
    async def _get_llm_response(self, chat_id: int, messages: List[Dict[str, str]]) -> str:
        """Get response from OpenRouter API."""
        if not self.openrouter_api_key:
            self.logger.error(f"[ChatID: {chat_id}] OpenRouter API Key is not configured!")
            return "*error beep* My API key is not set up!"
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "HTTP-Referer": "https://Kovuai.com",
                "X-Title": "Kovu Telegram Bot"
            }
            
            payload = {
                "model": self.OPENROUTER_MODEL,
                "messages": messages,
                "max_tokens": self.OPENROUTER_DEFAULT_MAX_TOKENS_ASK,
                "temperature": 0.5,
                "presence_penalty": 0.3,
                "frequency_penalty": 0.3,
                "top_p": 0.9,
                "stream": False
            }
            
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(
                    self.OPENROUTER_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=self.OPENROUTER_TIMEOUT_SECONDS
                ) as response:
                    response.raise_for_status()
                    response_json = await response.json()
                    
                    if (response_json.get("choices") and
                        len(response_json["choices"]) > 0 and
                        response_json["choices"][0].get("message") and
                        response_json["choices"][0]["message"].get("content")):
                        return response_json["choices"][0]["message"]["content"].strip()
                    else:
                        self.logger.error(f"[ChatID: {chat_id}] Invalid LLM response format: {response_json}")
                        return "*error beep* My circuits are fuzzy... try again later!"
                        
        except asyncio.TimeoutError as e:
            self.logger.error(f"[ChatID: {chat_id}] API request timed out: {e}")
            return "*system whine* My processors are overloaded! Try again soon."
        except aiohttp.ClientResponseError as e:
            self.logger.error(f"[ChatID: {chat_id}] API HTTP Error: {e.status} - {e.message}")
            if e.status == 401:
                return "*critical error* My API key is invalid!"
            elif e.status == 429:
                return "*system whine* Too many requests! Please wait a moment."
            return "*error beep* An API error occurred. Try again later."
        except Exception as e:
            self.logger.error(f"[ChatID: {chat_id}] Unexpected error: {e}")
            return "*error beep* An unexpected error occurred."
    
    def _prepare_messages_for_llm(self, chat_id: int, user_question: str = None) -> List[Dict[str, str]]:
        """Prepare messages for LLM API call."""
        messages = [{"role": "system", "content": self.KOVU_SYSTEM_PROMPT}]
        
        if user_question:
            messages.append({"role": "user", "content": user_question})
        elif chat_id in self.chat_rooms and self.chat_rooms[chat_id].message_history:
            # Get recent messages for context
            recent_messages = self.chat_rooms[chat_id].message_history[-20:]
            
            for msg in recent_messages:
                if msg.get('role') == 'user' and msg.get('content', '').strip():
                    username = msg.get('username', 'Unknown')
                    content = f"from: @{username}: {msg['content']}" if username != 'Unknown' else msg['content']
                    messages.append({"role": "user", "content": content})
                elif msg.get('role') == 'assistant' and msg.get('content', '').strip():
                    messages.append({"role": "assistant", "content": msg['content']})
        
        return messages
    
    async def _periodic_llm_response(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle periodic LLM responses."""
        job = context.job
        chat_id = job.data['chat_id']
        
        self.logger.info(f"[ChatID: {chat_id}] Checking for periodic LLM response...")
        
        # Skip if no new activity
        if not self.chat_rooms.get(chat_id, ChatData()).new_activity_flag:
            self.logger.info(f"[ChatID: {chat_id}] No new activity - skipping LLM call")
            return
        
        try:
            messages = self._prepare_messages_for_llm(chat_id)
            
            # Ensure we have user messages to respond to
            if not any(msg['role'] == 'user' for msg in messages[1:]):  # Skip system prompt
                messages.append({
                    "role": "user",
                    "content": "Hello Kovu! Please introduce yourself to the group chat."
                })
            
            # Make API call with periodic-specific settings
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "HTTP-Referer": "https://Kovuai.com",
                "X-Title": "Kovu Telegram Bot"
            }
            
            payload = {
                "model": self.OPENROUTER_MODEL,
                "messages": messages,
                "max_tokens": self.OPENROUTER_DEFAULT_MAX_TOKENS_PERIODIC,
                "temperature": 0.7,
                "presence_penalty": 0.3,
                "frequency_penalty": 0.3,
                "top_p": 0.9,
                "stream": False
            }
            
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(
                    self.OPENROUTER_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=self.OPENROUTER_PERIODIC_TIMEOUT_SECONDS
                ) as response:
                    response.raise_for_status()
                    response_json = await response.json()
            
            if (response_json.get("choices") and
                len(response_json["choices"]) > 0 and
                response_json["choices"][0].get("message") and
                response_json["choices"][0]["message"].get("content")):
                
                response_text = response_json["choices"][0]["message"]["content"].strip()
                
                if response_text:
                    await context.bot.send_message(chat_id=chat_id, text=response_text)
                    self._add_message_to_history(chat_id, "assistant", response_text, 
                                               context.bot.id, "kovuaibot")
                    # Reset activity flag after successful response
                    self.chat_rooms[chat_id].new_activity_flag = False
                    self.logger.info(f"[ChatID: {chat_id}] Sent periodic response and reset activity flag")
                    
        except Exception as e:
            self.logger.error(f"[ChatID: {chat_id}] Error in periodic LLM response: {e}")
    
    async def handle_ask_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /ask command."""
        if not update.message or not update.message.text:
            return
        
        question = update.message.text.replace('/ask', '').strip()
        if not question:
            await update.message.reply_text("Usage: /ask [your question]")
            return
        
        chat_id = update.effective_chat.id
        self._ensure_chat_data(chat_id)
        
        self.logger.info(f"[ChatID: {chat_id}] Processing /ask command: '{question}'")
        
        # Prepare question with user context
        user_context = f"from: @{update.message.from_user.username}: {question}" if update.message.from_user.username else question
        messages = self._prepare_messages_for_llm(chat_id, user_context)
        
        reply = await self._get_llm_response(chat_id, messages)
        await update.message.reply_text(reply)
        
        # Update history
        user_id = update.message.from_user.id
        username = update.message.from_user.username or update.message.from_user.first_name or "Unknown"
        self._add_message_to_history(chat_id, "user", question, user_id, username)
        self._add_message_to_history(chat_id, "assistant", reply, context.bot.id, "kovuaibot")
    
    async def handle_start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command and set up periodic job."""
        chat_id = update.effective_chat.id
        self._ensure_chat_data(chat_id)
        
        # Set activity flag to encourage immediate response
        self.chat_rooms[chat_id].new_activity_flag = True
        
        self.logger.info(f"[ChatID: {chat_id}] Setting up periodic LLM job")
        
        if not hasattr(context.application, 'job_queue') or not isinstance(context.application.job_queue, JobQueue):
            await update.message.reply_text("Error: Could not access the task scheduler.")
            return
        
        job_queue = context.application.job_queue
        job_name = f"llm_response_job_{chat_id}"
        
        # Remove existing jobs
        current_jobs = job_queue.get_jobs_by_name(job_name)
        for job in current_jobs:
            job.schedule_removal()
            self.logger.info(f"[ChatID: {chat_id}] Removed existing job: {job.name}")
        
        # Schedule new job
        job_queue.run_repeating(
            callback=self._periodic_llm_response,
            interval=self.PERIODIC_JOB_INTERVAL_SECONDS,
            first=self.PERIODIC_JOB_FIRST_RUN_DELAY_SECONDS,
            name=job_name,
            data={'chat_id': chat_id}
        )
        
        self.logger.info(f"[ChatID: {chat_id}] Scheduled periodic job with {self.PERIODIC_JOB_INTERVAL_SECONDS}s interval")
        await update.message.reply_text("I'm now set up to think and respond periodically! Talk to me!")
    
    async def handle_new_members(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Welcome new chat members."""
        if not update.message or not update.message.new_chat_members:
            return
        
        chat_id = update.effective_chat.id
        self._ensure_chat_data(chat_id)
        
        for member in update.message.new_chat_members:
            if member.is_bot:
                continue
                
            welcome_message = f"Welcome to the Kovu AI CTO, {member.full_name}! Glad to have you here.."
            self.logger.info(f"[ChatID: {chat_id}] Welcoming new member: {member.full_name}")
            await update.message.reply_text(welcome_message)
            
            self._add_message_to_history(chat_id, "assistant", welcome_message, 
                                       context.bot.id, "kovuaibot")
    
    async def handle_text_messages(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming text messages."""
        if not update.message or not update.message.text:
            return
        
        chat_id = update.effective_chat.id
        user_id = update.message.from_user.id
        username = (update.message.from_user.username or 
                   update.message.from_user.first_name or "Unknown")
        
        self.logger.info(f"[ChatID: {chat_id}] Message from @{username}: '{update.message.text}'")
        
        self._ensure_chat_data(chat_id)
        self._add_message_to_history(chat_id, "user", update.message.text, user_id, username)
        
        # Handle potential keyword responses (currently disabled in original code)
        # await self._handle_keywords(update, context)
    
    def _setup_handlers(self) -> None:
        """Set up message and command handlers."""
        if not self.application:
            raise RuntimeError("Application not initialized")
        
        self.application.add_handler(CommandHandler("start", self.handle_start_command))
        self.application.add_handler(CommandHandler("ask", self.handle_ask_command))
        self.application.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, self.handle_new_members))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text_messages))
    
    def run(self) -> None:
        """Start the bot."""
        self.logger.info("Initializing Kovu Bot...")
        
        self.application = Application.builder().token(self.bot_token).build()
        self._setup_handlers()
        
        self.logger.info("Starting bot polling...")
        try:
            self.application.run_polling(allowed_updates=Update.ALL_TYPES)
        except Exception as e:
            self.logger.critical(f"Bot polling failed: {e}", exc_info=True)
            raise
        finally:
            self.logger.info("Bot polling stopped.")


def main() -> None:
    """Main entry point."""
    try:
        bot = KovuBot()
        bot.run()
    except Exception as e:
        logging.getLogger(__name__).critical(f"Failed to start bot: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()