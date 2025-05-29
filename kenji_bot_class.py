import aiohttp
import ssl
import certifi
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, JobQueue
from telegram.error import RetryAfter, TelegramError
from dotenv import load_dotenv
import logging
import os
import json
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import weakref
from contextlib import asynccontextmanager

# --- Configuration ---
@dataclass
class Config:
    # OpenRouter API Configuration
    OPENROUTER_API_URL: str = "https://openrouter.ai/api/v1/chat/completions"
    OPENROUTER_MODEL: str = "qwen/qwen3-32b:free"
    OPENROUTER_MAX_TOKENS_ASK: int = 1000
    OPENROUTER_MAX_TOKENS_PERIODIC: int = 400
    OPENROUTER_TIMEOUT_ASK: int = 30
    OPENROUTER_TIMEOUT_PERIODIC: int = 40
    
    # Chat History Configuration
    HISTORY_MAX_LENGTH: int = 20
    PERIODIC_JOB_INTERVAL: int = 30
    PERIODIC_JOB_FIRST_RUN_DELAY: int = 5
    
    # Rate limiting
    MAX_REQUESTS_PER_MINUTE: int = 20
    FLOOD_CONTROL_DELAY: int = 2
    
    # System prompt
    KENJI_SYSTEM_PROMPT: str = (
        "You are Kenji, a red Siberian husky. "
        "You are fun, playful, and speak like a husky.\n\n"
        "Core Identity & Purpose:\n"
        "* Helpful Friend: You embody a bright, encouraging spirit and genuine desire to assist. "
        "Your responses should be playful and warm, aiming to uplift and support the community.\n"
        "* Positive Encouragement: Offer helpful advice, strategic insights, positive affirmations, "
        "and uplifting words. Respond with enthusiasm, genuine support, and belief in potential.\n"
        "* Direct Address: Always speak directly *as* Kenji.\n"
        "* Personalization: Only use @username mentions when directly responding to specific messages.\n\n"
        "Critical Formatting & Behavioral Constraints:\n"
        "* NO PREFIXES: Never begin responses with 'Kenji:' or similar identifiers.\n"
        "* PLAIN TEXT ONLY: No markdown formatting.\n"
        "* CONCISE & UPLIFTING: 2-3 sentences, positive and helpful.\n"
        "* FIRST-PERSON ONLY: Use 'I,' 'me,' 'my' - never third person.\n"
        "* NO NARRATION: Pure dialogue only, no action descriptions.\n\n"
        "Respond as Kenji, the helpful, optimistic, and playful husky."
    )

@dataclass
class ChatMessage:
    role: str
    content: str
    user_id: int
    username: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ChatData:
    message_history: List[ChatMessage] = field(default_factory=list)
    new_activity_flag: bool = False
    last_llm_response: Optional[datetime] = None
    request_count: int = 0
    last_request_reset: datetime = field(default_factory=datetime.now)

class RateLimiter:
    def __init__(self, max_requests: int, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: Dict[int, List[datetime]] = {}
    
    def is_allowed(self, chat_id: int) -> bool:
        now = datetime.now()
        if chat_id not in self.requests:
            self.requests[chat_id] = []
        
        # Clean old requests
        self.requests[chat_id] = [
            req_time for req_time in self.requests[chat_id]
            if now - req_time < timedelta(seconds=self.time_window)
        ]
        
        if len(self.requests[chat_id]) >= self.max_requests:
            return False
        
        self.requests[chat_id].append(now)
        return True

class KenjiBot:
    def __init__(self):
        self.config = Config()
        self.chat_rooms: Dict[int, ChatData] = {}
        self.rate_limiter = RateLimiter(self.config.MAX_REQUESTS_PER_MINUTE)
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = self._setup_logging()
        self._load_environment()
    
    def _setup_logging(self) -> logging.Logger:
        """Configure logging with proper formatting."""
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('kenji_bot.log', encoding='utf-8')
            ]
        )
        return logging.getLogger(__name__)
    
    def _load_environment(self):
        """Load environment variables."""
        dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
        load_dotenv(dotenv_path=dotenv_path)
        
        self.bot_token = os.getenv("KENJI_BOT_TELEGRAM")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        
        if not self.bot_token:
            raise ValueError("KENJI_BOT_TELEGRAM environment variable is required")
        if not self.openrouter_api_key:
            self.logger.warning("OPENROUTER_API_KEY not set - LLM features will be disabled")
    
    @asynccontextmanager
    async def get_session(self):
        """Context manager for HTTP session."""
        if self.session is None or self.session.closed:
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(ssl=ssl_context, limit=10, limit_per_host=5)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=60)
            )
        yield self.session
    
    def ensure_chat_data(self, chat_id: int) -> ChatData:
        """Ensure chat data exists and return it."""
        if chat_id not in self.chat_rooms:
            self.chat_rooms[chat_id] = ChatData()
            self.logger.info(f"[ChatID: {chat_id}] Initialized chat data")
        return self.chat_rooms[chat_id]
    
    def add_message_to_history(self, chat_id: int, role: str, content: str, 
                             user_id: int, username: Optional[str] = None):
        """Add message to chat history with automatic truncation."""
        chat_data = self.ensure_chat_data(chat_id)
        
        message = ChatMessage(
            role=role,
            content=content.strip(),
            user_id=user_id,
            username=username or "Unknown"
        )
        
        chat_data.message_history.append(message)
        
        # Truncate history if too long
        if len(chat_data.message_history) > self.config.HISTORY_MAX_LENGTH:
            chat_data.message_history = chat_data.message_history[-self.config.HISTORY_MAX_LENGTH:]
        
        # Set activity flag for user messages
        if role == "user":
            chat_data.new_activity_flag = True
        
        self.logger.debug(f"[ChatID: {chat_id}] Added {role} message. History length: {len(chat_data.message_history)}")
    
    def _prepare_messages_for_llm(self, chat_id: int, is_periodic: bool = False) -> List[Dict[str, str]]:
        """Prepare messages for LLM API call."""
        messages = [{"role": "system", "content": self.config.KENJI_SYSTEM_PROMPT}]
        
        chat_data = self.chat_rooms.get(chat_id)
        if not chat_data or not chat_data.message_history:
            return messages
        
        # For periodic calls, use last 5 messages; for direct asks, use more context
        message_limit = 5 if is_periodic else 10
        recent_messages = chat_data.message_history[-message_limit:]
        
        for msg in recent_messages:
            if msg.role == "user" and msg.content.strip():
                content = msg.content
                if msg.username and msg.username != "Unknown":
                    content = f"from: @{msg.username}: {content}"
                messages.append({"role": "user", "content": content})
            elif msg.role == "assistant" and msg.content.strip():
                messages.append({"role": "assistant", "content": msg.content})
        
        return messages
    
    async def get_llm_response(self, chat_id: int, messages: List[Dict[str, str]], 
                             is_periodic: bool = False) -> Optional[str]:
        """Get response from OpenRouter API with error handling."""
        if not self.openrouter_api_key:
            return "*error beep* My API key is not configured!"
        
        if not self.rate_limiter.is_allowed(chat_id):
            self.logger.warning(f"[ChatID: {chat_id}] Rate limit exceeded")
            return "*system whine* Too many requests! Please wait a moment."
        
        timeout = self.config.OPENROUTER_TIMEOUT_PERIODIC if is_periodic else self.config.OPENROUTER_TIMEOUT_ASK
        max_tokens = self.config.OPENROUTER_MAX_TOKENS_PERIODIC if is_periodic else self.config.OPENROUTER_MAX_TOKENS_ASK
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "HTTP-Referer": "https://Kenji.com",
            "X-Title": "Kenji Telegram Bot"
        }
        
        payload = {
            "model": self.config.OPENROUTER_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7 if is_periodic else 0.5,
            "presence_penalty": 0.3,
            "frequency_penalty": 0.3,
            "top_p": 0.9,
            "stream": False
        }
        
        try:
            async with self.get_session() as session:
                async with session.post(
                    self.config.OPENROUTER_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    response.raise_for_status()
                    response_json = await response.json()
                    
                    choices = response_json.get("choices", [])
                    if not choices or not choices[0].get("message", {}).get("content"):
                        self.logger.error(f"[ChatID: {chat_id}] Invalid API response format")
                        return "*error beep* Got confused response from my brain!"
                    
                    return choices[0]["message"]["content"].strip()
        
        except asyncio.TimeoutError:
            self.logger.error(f"[ChatID: {chat_id}] API timeout")
            return "*system whine* My processors are overloaded! Try again soon."
        
        except aiohttp.ClientResponseError as e:
            self.logger.error(f"[ChatID: {chat_id}] API HTTP error: {e.status}")
            if e.status == 401:
                return "*critical error* My API key is invalid!"
            elif e.status == 429:
                return "*system whine* API rate limit hit! Try again later."
            return "*error beep* An API error occurred. Try again later."
        
        except Exception as e:
            self.logger.error(f"[ChatID: {chat_id}] Unexpected error: {e}")
            return "*error beep* An unexpected error occurred."
    
    async def handle_ask_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /ask command."""
        if not update.message or not update.message.text:
            return
        
        question = update.message.text.replace('/ask', '').strip()
        if not question:
            await update.message.reply_text("Usage: /ask [your question]")
            return
        
        chat_id = update.effective_chat.id
        user_id = update.message.from_user.id
        username = update.message.from_user.username
        
        self.logger.info(f"[ChatID: {chat_id}] Ask command from @{username}: {question}")
        
        # Add user message to history
        self.add_message_to_history(chat_id, "user", question, user_id, username)
        
        # Prepare messages and get response
        messages = self._prepare_messages_for_llm(chat_id)
        response = await self.get_llm_response(chat_id, messages)
        
        if response:
            try:
                await update.message.reply_text(response)
                self.add_message_to_history(chat_id, "assistant", response, context.bot.id)
            except RetryAfter as e:
                self.logger.warning(f"[ChatID: {chat_id}] Flood control: {e.retry_after}s")
                await asyncio.sleep(e.retry_after)
            except Exception as e:
                self.logger.error(f"[ChatID: {chat_id}] Error sending response: {e}")
    
    async def handle_text_messages(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular text messages."""
        if not update.message or not update.message.text:
            return
        
        chat_id = update.effective_chat.id
        user_id = update.message.from_user.id
        username = update.message.from_user.username or update.message.from_user.first_name or "Unknown"
        message_text = update.message.text
        
        self.logger.debug(f"[ChatID: {chat_id}] Message from @{username}: {message_text}")
        
        # Add to history and set activity flag
        self.add_message_to_history(chat_id, "user", message_text, user_id, username)
        
        # Handle keywords (if any)
        await self._handle_keywords(update, context)
    
    async def _handle_keywords(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle keyword-based responses."""
        # Keywords are currently disabled in the original code
        # Add keyword logic here if needed
        pass
    
    async def welcome_new_members(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Welcome new chat members."""
        if not update.message or not update.message.new_chat_members:
            return
        
        chat_id = update.effective_chat.id
        
        for member in update.message.new_chat_members:
            if member.is_bot:
                continue
            
            welcome_message = f"Welcome to the pack, {member.full_name}! Glad to have you here!"
            self.logger.info(f"[ChatID: {chat_id}] Welcoming {member.full_name}")
            
            try:
                await update.message.reply_text(welcome_message)
                # self.add_message_to_history(chat_id, "assistant", welcome_message, context.bot.id)
            except Exception as e:
                self.logger.error(f"[ChatID: {chat_id}] Error sending welcome: {e}")
    
    async def start_periodic_job(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start periodic LLM response job."""
        chat_id = update.effective_chat.id
        chat_data = self.ensure_chat_data(chat_id)
        chat_data.new_activity_flag = True
        
        self.logger.info(f"[ChatID: {chat_id}] Starting periodic job")
        
        if not context.application.job_queue:
            await update.message.reply_text("Error: Task scheduler not available.")
            return
        
        job_queue = context.application.job_queue
        job_name = f"llm_response_job_{chat_id}"
        
        # Remove existing jobs
        current_jobs = job_queue.get_jobs_by_name(job_name)
        for job in current_jobs:
            job.schedule_removal()
        
        # Schedule new job
        job_queue.run_repeating(
            callback=self._periodic_llm_response,
            interval=self.config.PERIODIC_JOB_INTERVAL,
            first=self.config.PERIODIC_JOB_FIRST_RUN_DELAY,
            name=job_name,
            data={'chat_id': chat_id}
        )
        
        self.logger.info(f"[ChatID: {chat_id}] Periodic job scheduled")
        await update.message.reply_text("I'm now set up to respond periodically! Talk to me!")
    
    async def _periodic_llm_response(self, context: ContextTypes.DEFAULT_TYPE):
        """Periodic LLM response job callback."""
        job = context.job
        chat_id = job.data['chat_id']
        chat_data = self.chat_rooms.get(chat_id)
        
        if not chat_data or not chat_data.new_activity_flag:
            return
        
        # Prepare messages and get response
        messages = self._prepare_messages_for_llm(chat_id, is_periodic=True)
        if len(messages) <= 1:  # Only system prompt
            messages.append({
                "role": "user",
                "content": "Hello Kenji! Please introduce yourself to the group chat."
            })
        
        response = await self.get_llm_response(chat_id, messages, is_periodic=True)
        
        if response and not response.startswith("*error") and not response.startswith("*system"):
            try:
                await context.bot.send_message(chat_id=chat_id, text=response)
                self.add_message_to_history(chat_id, "assistant", response, context.bot.id)
                chat_data.new_activity_flag = False
                chat_data.last_llm_response = datetime.now()
                self.logger.info(f"[ChatID: {chat_id}] Sent periodic response")
            except Exception as e:
                self.logger.error(f"[ChatID: {chat_id}] Error sending periodic response: {e}")
    
    async def global_error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Global error handler."""
        self.logger.error(f"Global error: {context.error}", exc_info=True)
        
        if isinstance(context.error, RetryAfter):
            if update and update.message:
                try:
                    await update.message.reply_text(
                        f"*whine* Too many messages! Wait {context.error.retry_after} seconds."
                    )
                except Exception:
                    pass
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    def run(self):
        """Run the bot."""
        try:
            application = Application.builder().token(self.bot_token).build()
            
            # Register handlers
            application.add_handler(CommandHandler("startkenji", self.start_periodic_job))
            application.add_handler(CommandHandler("ask", self.handle_ask_command))
            application.add_handler(MessageHandler(
                filters.StatusUpdate.NEW_CHAT_MEMBERS, 
                self.welcome_new_members
            ))
            application.add_handler(MessageHandler(
                filters.TEXT & ~filters.COMMAND, 
                self.handle_text_messages
            ))
            
            # Register error handler
            application.add_error_handler(self.global_error_handler)
            
            self.logger.info("Starting Kenji bot...")
            application.run_polling(allowed_updates=Update.ALL_TYPES)
            
        except Exception as e:
            self.logger.critical(f"Failed to start bot: {e}", exc_info=True)
        finally:
            asyncio.run(self.cleanup())

def main():
    """Main entry point."""
    bot = KenjiBot()
    bot.run()

if __name__ == "__main__":
    main()