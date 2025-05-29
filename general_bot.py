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
import argparse
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import weakref


@dataclass
class ChatData:
    """Data structure for chat room information."""
    message_history: List[Dict[str, Any]] = field(default_factory=list)
    last_processed_history_len: int = 0
    new_activity_flag: bool = False
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class BotConfig:
    """Configuration data for the bot."""
    name: str
    bot_name_display: str
    bot_username_internal: str
    env_prefix: str
    system_prompt_file: str
    openrouter_model: str
    welcome_message_template: str = "Welcome, {member_full_name}!"
    telegram_bot_token: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    system_prompt: Optional[str] = None


class TelegramBot:
    """
    A class-based Telegram bot with LLM integration and periodic responses.
    """
    
    # Class constants
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
    DEFAULT_MAX_TOKENS_ASK = 1000
    DEFAULT_MAX_TOKENS_PERIODIC = 600
    TIMEOUT_SECONDS = 30
    PERIODIC_TIMEOUT_SECONDS = 60
    HISTORY_MAX_LENGTH = 20
    PERIODIC_JOB_INTERVAL_SECONDS = 30
    PERIODIC_JOB_FIRST_RUN_DELAY_SECONDS = 5
    
    def __init__(self, bot_name: str):
        """Initialize the bot with configuration."""
        self.bot_name = bot_name
        self.config: Optional[BotConfig] = None
        self.chat_rooms: Dict[int, ChatData] = {}
        self.application: Optional[Application] = None
        self.logger = logging.getLogger(f"{bot_name}")
        self._ssl_context = ssl.create_default_context(cafile=certifi.where())
        
    def setup_logging(self) -> None:
        """Configure logging for the bot."""
        log_level_str = os.getenv(f"{self.config.env_prefix}LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        
        logging.basicConfig(
            format=f'%(asctime)s - [{self.config.bot_name_display}] - %(levelname)s - %(message)s',
            level=log_level
        )
        
        # Suppress verbose library logging
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("telegram").setLevel(logging.INFO)
        logging.getLogger("apscheduler").setLevel(logging.WARNING)
        
    def load_config(self) -> bool:
        """Load bot configuration from JSON file."""
        try:
            config_path = os.path.join(os.path.dirname(__file__), "bots_config.json")
            with open(config_path, 'r') as f:
                all_configs = json.load(f)
            
            if self.bot_name not in all_configs:
                self.logger.critical(f"Bot configuration for '{self.bot_name}' not found.")
                return False
            
            config_data = all_configs[self.bot_name]
            config_data['name'] = self.bot_name
            self.config = BotConfig(**config_data)
            return True
            
        except FileNotFoundError:
            self.logger.critical("bots_config.json not found.")
            return False
        except json.JSONDecodeError:
            self.logger.critical("Invalid JSON in bots_config.json.")
            return False
        except TypeError as e:
            self.logger.critical(f"Invalid configuration structure: {e}")
            return False
    
    def load_environment_variables(self) -> bool:
        """Load environment variables."""
        dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
        load_dotenv(dotenv_path=dotenv_path)
        
        env_prefix = self.config.env_prefix
        self.config.telegram_bot_token = os.getenv(f"{env_prefix}BOT_TOKEN")
        self.config.openrouter_api_key = (
            os.getenv(f"{env_prefix}OPENROUTER_API_KEY") or 
            os.getenv("OPENROUTER_API_KEY")
        )
        
        if not self.config.telegram_bot_token:
            self.logger.critical(f"BOT_TOKEN not set (expected: {env_prefix}_BOT_TOKEN)")
            return False
        
        if not self.config.openrouter_api_key:
            self.logger.warning("OPENROUTER_API_KEY not set. LLM features may fail.")
        
        return True
    
    def load_system_prompt(self) -> bool:
        """Load system prompt from file."""
        prompt_file_path = self.config.system_prompt_file
        
        if not os.path.isabs(prompt_file_path):
            prompt_file_path = os.path.join(os.path.dirname(__file__), prompt_file_path)
        
        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                self.config.system_prompt = f.read().strip()
            self.logger.info(f"System prompt loaded from: {prompt_file_path}")
            return True
        except FileNotFoundError:
            self.logger.critical(f"System prompt file not found: {prompt_file_path}")
            return False
        except Exception as e:
            self.logger.critical(f"Error loading system prompt: {e}")
            return False
    
    def ensure_chat_data(self, chat_id: int) -> ChatData:
        """Ensure chat data exists and return it."""
        if chat_id not in self.chat_rooms:
            self.chat_rooms[chat_id] = ChatData()
            self.logger.info(f"[ChatID: {chat_id}] Initialized chat data.")
        return self.chat_rooms[chat_id]
    
    def add_message_to_history(self, chat_id: int, role: str, content: str, 
                             user_id: int, username: str = "Unknown", 
                             is_reply_to_user: bool = False) -> None:
        """Add a message to chat history."""
        chat_data = self.ensure_chat_data(chat_id)
        
        message = {
            "role": role,
            "content": content,
            "user_id": user_id,
            "username": username,
            "timestamp": datetime.now().isoformat(),
            "is_reply_to_user": is_reply_to_user
        }
        
        chat_data.message_history.append(message)
        chat_data.message_history = chat_data.message_history[-self.HISTORY_MAX_LENGTH:]
        
        # Only set new_activity_flag if it's not a user replying to another user
        if not is_reply_to_user:
            chat_data.new_activity_flag = True
        
        self.logger.info(f"[ChatID: {chat_id}] Message added (role: {role}, reply_to_user: {is_reply_to_user}). "
                        f"History length: {len(chat_data.message_history)}")
    
    async def get_llm_response(self, chat_id: int, messages: List[Dict], 
                             for_periodic_job: bool = False) -> str:
        """Get response from LLM API."""
        if not self.config.openrouter_api_key:
            self.logger.error(f"[ChatID: {chat_id}] OpenRouter API Key not configured.")
            return "*error beep* My API key is not set up!"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.openrouter_api_key}",
            "HTTP-Referer": f"https://{self.config.bot_username_internal}.com",
            "X-Title": f"{self.config.bot_name_display} Telegram Integration"
        }
        
        payload = {
            "model": self.config.openrouter_model,
            "messages": messages,
            "max_tokens": (self.DEFAULT_MAX_TOKENS_PERIODIC if for_periodic_job 
                          else self.DEFAULT_MAX_TOKENS_ASK),
            "temperature": 0.7 if for_periodic_job else 0.5,
            "presence_penalty": 0.3,
            "frequency_penalty": 0.3,
            "stream": False
        }
        
        if for_periodic_job:
            payload["top_p"] = 0.9
        
        timeout = (self.PERIODIC_TIMEOUT_SECONDS if for_periodic_job 
                  else self.TIMEOUT_SECONDS)
        
        try:
            connector = aiohttp.TCPConnector(ssl=self._ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(
                    self.OPENROUTER_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    response.raise_for_status()
                    response_json = await response.json()
                    self.logger.info(f"[LLM DEBUG] Full LLM API response: {response_json}")
                    choices = response_json.get("choices", [])
                    if choices and choices[0].get("message", {}).get("content"):
                        content = choices[0]["message"]["content"].strip()
                        self.logger.info(f"[LLM DEBUG] Outgoing LLM message: '{content}' (length: {len(content)})")
                        return content
                    else:
                        self.logger.error(f"[ChatID: {chat_id}] Unexpected LLM response format. Full response: {response_json}")
                        return "*error beep* My circuits are fuzzy... try again later!"
        
        except asyncio.TimeoutError:
            self.logger.error(f"[ChatID: {chat_id}] LLM API request timed out")
            return "*system whine* My processors are overloaded! Try again soon."
        except aiohttp.ClientResponseError as e:
            self.logger.error(f"[ChatID: {chat_id}] LLM API HTTP Error: {e.status}")
            if e.status == 401:
                return "*critical error* My API key is invalid!"
            return "*error beep* An API error occurred. Try again later."
        except Exception as e:
            self.logger.error(f"[ChatID: {chat_id}] LLM Response Error: {e}")
            return "*error beep* An unexpected error occurred."
    
    def prepare_messages_for_llm(self, chat_id: int, 
                               include_introduction: bool = False) -> List[Dict]:
        """Prepare messages for LLM API call."""
        messages = [{"role": "system", "content": self.config.system_prompt}]
        
        chat_data = self.chat_rooms.get(chat_id)
        if not chat_data or not chat_data.message_history:
            if include_introduction:
                messages.append({
                    "role": "user",
                    "content": f"Hello {self.config.bot_name_display}! Please introduce yourself to the group chat."
                })
            return messages
        
        # Process recent messages
        recent_messages = chat_data.message_history[-self.HISTORY_MAX_LENGTH:]
        valid_messages = []
        
        for msg in recent_messages:
            content = msg.get('content', '').strip()
            if not content:
                continue
                
            if msg.get('role') == 'user':
                username = msg.get('username', 'Unknown')
                formatted_content = (f"from: @{username}: {content}" 
                                   if username != "Unknown" else content)
                valid_messages.append({"role": "user", "content": formatted_content})
            elif msg.get('role') == 'assistant':
                valid_messages.append({"role": "assistant", "content": content})
        
        if not any(m['role'] == 'user' for m in valid_messages):
            if include_introduction:
                messages.append({
                    "role": "user", 
                    "content": f"Hello {self.config.bot_name_display}! Please introduce yourself."
                })
            return messages
        
        messages.extend(valid_messages)
        return messages
    
    def _get_last_active_user(self, chat_id: int) -> str:
        """Get the username of the last active user in the chat."""
        chat_data = self.chat_rooms.get(chat_id)
        if not chat_data or not chat_data.message_history:
            return None
        
        # Look for the most recent user message
        for msg in reversed(chat_data.message_history):
            if msg.get('role') == 'user' and msg.get('username'):
                return msg.get('username')
        
        return None
    
    def _is_user_replying_to_user(self, update: Update) -> bool:
        """Check if the message is a user replying to another user (not the bot)."""
        if not update.message or not update.message.reply_to_message:
            return False
        
        # Check if the original message was from a user (not the bot)
        replied_to_message = update.message.reply_to_message
        bot_id = update.get_bot().id if update.get_bot() else None
        
        # If replying to the bot, it's not a user-to-user reply
        if replied_to_message.from_user and replied_to_message.from_user.id == bot_id:
            return False
        
        # If replying to another user (not bot), it's a user-to-user reply
        if replied_to_message.from_user and not replied_to_message.from_user.is_bot:
            return True
        
        return False
    
    async def periodic_llm_callback(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Callback for periodic LLM responses."""
        job = context.job
        chat_id = job.data['chat_id']
        
        self.logger.info(f"[ChatID: {chat_id}] Attempting periodic LLM response...")
        
        chat_data = self.chat_rooms.get(chat_id)
        if not chat_data or not chat_data.new_activity_flag:
            self.logger.info(f"[ChatID: {chat_id}] No new activity - skipping LLM call")
            return
        
        messages = self.prepare_messages_for_llm(chat_id, include_introduction=True)
        response_text = await self.get_llm_response(chat_id, messages, for_periodic_job=True)
        
        if (response_text and 
            not any(response_text.startswith(prefix) for prefix in ["*error", "*system", "*critical"])):
            
            # Check if we should tag the last active user
            last_user = self._get_last_active_user(chat_id)
            if last_user and last_user != "Unknown":
                # Only add @ tag if the response doesn't already contain it
                if f"@{last_user}" not in response_text:
                    response_text = f"@{last_user} {response_text}"
            
            # Log outgoing message content and length
            self.logger.info(f"[ChatID: {chat_id}] Outgoing periodic reply length: {len(response_text)} | Content: {repr(response_text)}")

            try:
                await context.bot.send_message(chat_id=chat_id, text=response_text)
            except Exception as e:
                self.logger.error(f"[ChatID: {chat_id}] Error sending periodic send_message: {e}")
                if len(response_text) < 100:
                    try:
                        await context.bot.send_message(chat_id=chat_id, text=f"`{response_text}`", parse_mode="Markdown")
                    except Exception as e2:
                        self.logger.error(f"[ChatID: {chat_id}] Error sending periodic as code block: {e2}")

            self.add_message_to_history(
                chat_id=chat_id,
                role="assistant", 
                content=response_text,
                user_id=context.bot.id,
                username=self.config.bot_username_internal
            )
            chat_data.new_activity_flag = False
            self.logger.info(f"[ChatID: {chat_id}] Sent periodic response and reset activity flag")
        elif response_text:
            self.logger.warning(f"[ChatID: {chat_id}] Periodic LLM returned error: {response_text}")
        else:
            self.logger.error(f"[ChatID: {chat_id}] Periodic LLM returned no response")
    
    async def handle_start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        chat_id = update.effective_chat.id
        chat_data = self.ensure_chat_data(chat_id)
        chat_data.new_activity_flag = True
        
        username = self._get_username(update.message.from_user)
        
        self.logger.info(f"[ChatID: {chat_id}] Received /start command from @{username}")
        
        if not context.application.job_queue:
            self.logger.error(f"[ChatID: {chat_id}] JobQueue not available")
            await update.message.reply_text(f"@{username} Error: Task scheduler not available.")
            return
        
        job_name = f"llm_response_job_{chat_id}_{self.config.bot_username_internal}"
        
        # Remove existing jobs
        existing_jobs = context.application.job_queue.get_jobs_by_name(job_name)
        for job in existing_jobs:
            job.schedule_removal()
            
        if existing_jobs:
            self.logger.info(f"[ChatID: {chat_id}] Removed {len(existing_jobs)} existing jobs")
        
        # Schedule new job
        context.application.job_queue.run_repeating(
            callback=self.periodic_llm_callback,
            interval=self.PERIODIC_JOB_INTERVAL_SECONDS,
            first=self.PERIODIC_JOB_FIRST_RUN_DELAY_SECONDS,
            name=job_name,
            data={'chat_id': chat_id}
        )
        
        self.logger.info(f"[ChatID: {chat_id}] Scheduled periodic LLM job")
        await update.message.reply_text(
            f"@{username} {self.config.bot_name_display} is now set up to think and respond periodically. Talk to me!"
        )
    
    async def handle_ask_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /ask command."""
        if not update.message or not update.message.text:
            return

        parts = update.message.text.split(" ", 1)
        username = self._get_username(update.message.from_user)

        if len(parts) < 2:
            await update.message.reply_text(
                f"@{username} Usage: /ask [your question to {self.config.bot_name_display}]"
            )
            return
        
        question = parts[1]
        chat_id = update.effective_chat.id
        user_id = update.message.from_user.id
        
        self.ensure_chat_data(chat_id)
        self.logger.info(f"[ChatID: {chat_id}] /ask command from @{username}: '{question}'")
        
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": f"from: @{username}: {question}"}
        ]
        
        reply = await self.get_llm_response(chat_id, messages)

        # Tag the user in the response
        if not reply.startswith(f"@{username}") and not any(reply.startswith(prefix) for prefix in ["*error", "*system", "*critical"]):
            reply = f"@{username} {reply}"

        # Log outgoing message content and length
        self.logger.info(f"[ChatID: {chat_id}] Outgoing reply length: {len(reply)} | Content: {repr(reply)}")

        try:
            await update.message.reply_text(reply)
        except Exception as e:
            self.logger.error(f"[ChatID: {chat_id}] Error sending reply_text: {e}")
            # As a test, try sending as a code block if short and failed
            if len(reply) < 100:
                try:
                    await update.message.reply_text(f"`{reply}`", parse_mode="Markdown")
                except Exception as e2:
                    self.logger.error(f"[ChatID: {chat_id}] Error sending as code block: {e2}")

        # Add to history
        self.add_message_to_history(chat_id, "user", question, user_id, username)
        if not any(reply.startswith(prefix) for prefix in ["*error", "*system", "*critical"]):
            self.add_message_to_history(
                chat_id, "assistant", reply, context.bot.id, self.config.bot_username_internal
            )
    
    async def handle_new_members(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle new chat members."""
        if not update.message or not update.message.new_chat_members:
            return
        
        chat_id = update.effective_chat.id
        self.ensure_chat_data(chat_id)
        
        for member in update.message.new_chat_members:
            if member.is_bot:
                self.logger.info(f"[ChatID: {chat_id}] Bot '{member.username}' joined - no welcome")
                continue
            
            welcome_message = self.config.welcome_message_template.format(
                member_full_name=member.full_name or member.username or "New Member"
            )
            
            # Tag the new member if they have a username
            if member.username:
                welcome_message = f"@{member.username} {welcome_message}"
            
            self.logger.info(f"[ChatID: {chat_id}] Welcoming: {member.full_name}")
            await update.message.reply_text(welcome_message)
            
    
    async def handle_text_messages(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle regular text messages."""
        if not update.message or not update.message.text:
            return
        
        chat_id = update.effective_chat.id
        user_id = update.message.from_user.id
        username = self._get_username(update.message.from_user)
        message_text = update.message.text
        
        # Check if this is a user replying to another user
        is_reply_to_user = self._is_user_replying_to_user(update)
        
        self.logger.info(f"[ChatID: {chat_id}] Message from @{username}: '{message_text}' "
                        f"(reply_to_user: {is_reply_to_user})")
        
        self.ensure_chat_data(chat_id)
        # Only add to history if not a reply to another user
        if not is_reply_to_user:
            self.add_message_to_history(chat_id, "user", message_text, user_id, username, is_reply_to_user)
    
    async def _handle_keywords(self, update: Update, context: ContextTypes.DEFAULT_TYPE, 
                             message_lower: str) -> None:
        """Handle keyword responses (extensible method)."""
        # Example keyword handling - can be extended or moved to config
        chat_id = update.effective_chat.id
        username = self._get_username(update.message.from_user)
        
        # Add custom keyword logic here
        if any(greeting in message_lower for greeting in ["good morning", "gm", "morning"]):
            response = f"@{username} Good morning to you too! 🌅"
            await update.message.reply_text(response)
            self.add_message_to_history(
                chat_id, "assistant", response, 
                context.bot.id, self.config.bot_username_internal
            )
    
    def _get_username(self, user) -> str:
        """Extract username from user object."""
        return (user.username or 
                getattr(user, 'first_name', None) or 
                getattr(user, 'full_name', None) or 
                "Unknown")
    
    async def handle_draw_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /draw command for image generation using OpenRouter multimodal model."""
        if not update.message or not update.message.text:
            return

        parts = update.message.text.split(" ", 1)
        username = self._get_username(update.message.from_user)

        if len(parts) < 2:
            await update.message.reply_text(
                f"@{username} Usage: /draw [description of the image you want]"
            )
            return

        prompt = parts[1]
        chat_id = update.effective_chat.id

        image_url = await self.generate_image_with_openrouter(prompt)
        if image_url:
            await update.message.reply_photo(photo=image_url, caption=f"@{username} Here is your image!")
        else:
            await update.message.reply_text(f"@{username} Sorry, I couldn't generate the image.")

    async def generate_image_with_openrouter(self, prompt: str) -> Optional[str]:
        """Call Modelslab image generation API and return the image URL."""
        api_url = "https://modelslab.com/api/v6/images/text2img"
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "key": "oyvfspVqJlBV2GNXc5rxnkNZm7Jgyuen4AA9xUTm4NeDMPEGntDfeg9sE7QB",  # Use the API key from config
            "prompt": prompt,
            "model_id": "tamarin-xl-v1",
            "samples": "1",
            "height": "1024",
            "width": "1024",
            "safety_checker": False,
            "seed": None,
            "base64": False,
            "webhook": None,
            "track_id": None,
            # "lora_model": "xl-realistic-cake-art-sty",
            # "lora_strength": "0.45",
        }
        try:
            connector = aiohttp.TCPConnector(ssl=self._ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    response.raise_for_status()
                    response_json = await response.json()
                    self.logger.info(f"[DRAW DEBUG] Modelslab response: {response_json}")
                    # The response should contain a 'output' field with a list of image URLs
                    output = response_json.get("output")
                    if output and isinstance(output, list) and len(output) > 0:
                        return output[0]
                    return None
        except Exception as e:
            self.logger.error(f"Image generation failed: {e}")
            return None
    
    def setup_handlers(self) -> None:
        """Setup message handlers."""
        if not self.application:
            raise RuntimeError("Application not initialized")
        
        handlers = [
            CommandHandler("start", self.handle_start_command),
            CommandHandler("ask", self.handle_ask_command),
            CommandHandler("draw", self.handle_draw_command),  # <-- Added draw handler
            MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, self.handle_new_members),
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text_messages),
        ]
        
        for handler in handlers:
            self.application.add_handler(handler)
        
        self.logger.info("Message handlers registered")
    
    def initialize(self) -> bool:
        """Initialize the bot with all configurations."""
        if not self.load_config():
            return False
        
        self.setup_logging()
        
        if not self.load_environment_variables():
            return False
        
        if not self.load_system_prompt():
            return False
        
        if not self.config.openrouter_model:
            self.logger.critical("OpenRouter model not configured")
            return False
        
        # Initialize Telegram application
        self.application = Application.builder().token(self.config.telegram_bot_token).build()
        self.setup_handlers()
        
        self.logger.info(f"Bot '{self.config.bot_name_display}' initialized successfully")
        return True
    
    def run(self) -> None:
        """Run the bot."""
        if not self.application:
            self.logger.critical("Bot not properly initialized")
            return
        
        self.logger.info(f"Starting bot polling for {self.config.bot_name_display}...")
        try:
            self.application.run_polling(allowed_updates=Update.ALL_TYPES)
        except KeyboardInterrupt:
            self.logger.info("Bot stopped by user")
        except Exception as e:
            self.logger.critical(f"Bot polling failed: {e}", exc_info=True)
        finally:
            self.logger.info("Bot polling stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run a Telegram Bot with specific personality.")
    parser.add_argument("--bot", required=True, 
                       help="Bot name (must match key in bots_config.json)")
    args = parser.parse_args()
    
    bot = TelegramBot(args.bot)
    
    if not bot.initialize():
        print(f"Failed to initialize bot '{args.bot}'", flush=True)
        return 1
    
    bot.run()
    return 0


if __name__ == "__main__":
    exit(main())