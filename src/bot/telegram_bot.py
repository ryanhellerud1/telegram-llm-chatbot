import aiohttp
import ssl
import certifi
from telegram import Update, error as telegram_error
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, JobQueue
from dotenv import load_dotenv
import logging
import os
import json
import asyncio
import argparse
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import async_timeout
from asyncio import Lock
from datetime import datetime
import weakref
import signal
import sys
from src.services.llm_client import LLMClient
from src.services.chat_data import ChatData, ChatHistoryManager
from src.config.config import BotConfig, ConfigLoader
from src.handlers.handlers import handle_start_command, handle_new_members, handle_text_messages
from src.handlers.handlers import handle_ask_command, handle_draw_command
from src.utils.helpers import get_username, is_user_replying_to_user


# Removed duplicate BotConfig definition


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
    _running = False  # Class variable to track running state
    _instances = weakref.WeakSet()  # Keep track of bot instances

    def __init__(self, bot_name: str):
        """Initialize the bot with configuration."""
        self.bot_name = bot_name
        self.config: Optional[BotConfig] = None
        self.application: Optional[Application] = None
        self.logger = logging.getLogger(f"{bot_name}")
        self._ssl_context = ssl.create_default_context(cafile=certifi.where())
        self._instances.add(self)  # Add this instance to the set
        self.llm_client = None  # Will be initialized after config is loaded
        self.chat_history_manager = ChatHistoryManager(history_max_length=self.HISTORY_MAX_LENGTH)
        self.periodic_jobs = {}  # Track periodic jobs per chat

    def setup_logging(self) -> None:
        """Configure logging for the bot."""
        log_level_str = os.getenv(f"{self.config.env_prefix}LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        
        # Get the root logger and set its level
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Create a formatter
        formatter = logging.Formatter(
            f'%(asctime)s - [{self.config.bot_name_display}] - %(levelname)s - %(message)s'
        )
        
        # Create a console handler if one doesn't exist
        console_handler = None
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                console_handler = handler
                break
        
        if not console_handler:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # Suppress verbose library logging
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("telegram").setLevel(logging.INFO)
        logging.getLogger("apscheduler").setLevel(logging.WARNING)
        
    def load_config(self) -> bool:
        """Load bot configuration from JSON file."""
        config = ConfigLoader.load_config(self.bot_name, logger=self.logger)
        if not config:
            return False
        self.config = config
        
        # Use absolute path for system prompt
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        prompt_file_path = os.path.join(base_dir, self.config.system_prompt_file)
            
        return True
    
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

        # Always resolve relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if not os.path.isabs(prompt_file_path):
            prompt_file_path = os.path.join(project_root, prompt_file_path)

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
        return self.chat_history_manager.ensure_chat_data(chat_id)
    
    def add_message_to_history(self, chat_id: int, role: str, content: str, 
                             user_id: int, username: str = "Unknown", 
                             is_reply_to_user: bool = False) -> None:
        """Add a message to chat history."""
        self.chat_history_manager.add_message_to_history(chat_id, role, content, user_id, username, is_reply_to_user, logger=self.logger)

        self.logger.info(f"[ChatID: {chat_id}] Message added (role: {role}, reply_to_user: {is_reply_to_user}). "
                        f"History length: {len(self.chat_history_manager.get_chat_data(chat_id).message_history)}")
    
    async def periodic_llm_callback(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        self.logger.info(f"[ChatID: {context.job.data['chat_id']}] periodic_llm_callback triggered (interval tick)")
        job = context.job
        chat_id = job.data['chat_id']
        
        self.logger.info(f"[ChatID: {chat_id}] Attempting periodic LLM response...")
        
        chat_data = self.chat_history_manager.get_chat_data(chat_id)
        if not chat_data or not chat_data.new_activity_flag:
            self.logger.info(f"[ChatID: {chat_id}] No new activity - skipping LLM call")
            return
        
        messages = self.prepare_messages_for_llm(chat_id, include_introduction=True)
        response_text = await self.llm_client.get_llm_response(chat_id, messages, for_periodic_job=True)
        
        if (response_text and 
            not any(response_text.startswith(prefix) for prefix in ["*error", "*system", "*critical"])):
            
            # Check if we should tag the last active user
            last_user = "Unknown"
            if chat_data and hasattr(chat_data, 'message_history'):
                for msg in reversed(chat_data.message_history):
                    if msg.get("role") == "user" and msg.get("username"):
                        last_user = msg["username"]
                        break
            if last_user != "Unknown" and not response_text.startswith(f"@{last_user}"):
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
    
    def setup_handlers(self) -> None:
        """Setup message handlers."""
        if not self.application:
            raise RuntimeError("Application not initialized")
        self.application.add_error_handler(self._handle_error)
        handlers = [
            CommandHandler("start", lambda update, context: handle_start_command(self, update, context)),
            CommandHandler("ask", lambda update, context: handle_ask_command(self, update, context)),
            CommandHandler("draw", lambda update, context: handle_draw_command(self, update, context)),
            MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, lambda update, context: handle_new_members(self, update, context)),
            MessageHandler(filters.TEXT & ~filters.COMMAND, lambda update, context: handle_text_messages(self, update, context)),
        ]
        for handler in handlers:
            self.application.add_handler(handler)
        self.logger.info("Message handlers registered")

    async def _handle_error(self, update: Optional[object], context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors in the bot."""
        if isinstance(context.error, telegram_error.Conflict):
            self.logger.error("Another instance detected, attempting graceful shutdown...")
            try:
                if self.application:
                    await self.application.stop()
                    await self.application.shutdown()
            except Exception as e:
                self.logger.error(f"Error during shutdown: {e}")
            finally:
                self._instances.discard(self)
                logging.shutdown()
                os._exit(1)
            return
        
        # Log any other errors
        self.logger.error(
            f"Exception while handling an update: {context.error}",
            exc_info=True
        )
    
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
        
        # Initialize LLMClient
        self.llm_client = LLMClient(
            api_key=self.config.openrouter_api_key,
            model=self.config.openrouter_model,
            bot_username_internal=self.config.bot_username_internal,
            bot_name_display=self.config.bot_name_display,
            logger=self.logger,
            ssl_context=self._ssl_context
        )
        # Initialize Telegram application
        self.application = Application.builder().token(self.config.telegram_bot_token).build()
        self.setup_handlers()
        
        self.logger.info(f"Bot '{self.config.bot_name_display}' initialized successfully")
        return True
    
    async def _check_telegram_api(self) -> bool:
        """Check if we can connect to Telegram API before starting."""
        try:
            connector = aiohttp.TCPConnector(ssl=self._ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(
                    f"https://api.telegram.org/bot{self.config.telegram_bot_token}/getMe",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 409:
                        return False
                    response.raise_for_status()
                    return True
        except Exception as e:
            self.logger.warning(f"API check failed: {e}")
            return False

    def run(self) -> None:
        """Run the bot."""
        if not self.application:
            self.logger.critical("Bot not properly initialized")
            return

        self.logger.info(f"Starting bot polling for {self.config.bot_name_display}...")
        try:
            # Clear any pending updates and start fresh
            self.application.run_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True,
                close_loop=False  # Don't close the loop to avoid asyncio errors
            )
        except KeyboardInterrupt:
            self.logger.info("Bot stopped by user")
        except Exception as e:
            self.logger.critical(f"Bot polling failed: {e}", exc_info=True)
        finally:
            self.logger.info("Bot polling stopped")
            try:
                if self.application and self.application.running:
                    self.application.stop_running()
                    self.application.shutdown()
            except Exception as e:
                self.logger.error(f"Error during shutdown: {e}")
            # Remove this instance from the tracked instances
            self._instances.discard(self)

    def prepare_messages_for_llm(self, chat_id: int, include_introduction: bool = False) -> list:
        """
        Prepare the message history for the LLM, optionally including a system introduction.
        Returns a list of dicts with 'role' and 'content'.
        """
        messages = []
        if include_introduction and self.config and hasattr(self.config, 'system_prompt'):
            messages.append({
                "role": "system",
                "content": self.config.system_prompt
            })
        chat_data = self.chat_history_manager.get_chat_data(chat_id)
        if chat_data and hasattr(chat_data, 'message_history'):
            for msg in chat_data.message_history:
                # msg is a dict, not an object
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        return messages
    
def signal_handler(sig, frame):
    print("Ctrl+C received, shutting down...", flush=True)
    sys.exit(0)

def main():
    """Main entry point."""
    # Set up logging to output to console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    signal.signal(signal.SIGINT, signal_handler)
    logger = logging.getLogger("MainProcess")
    
    logger.info("Starting bot initialization...")
    
    parser = argparse.ArgumentParser(description="Run a Telegram Bot with specific personality.")
    parser.add_argument("--bot", required=True, 
                       help="Bot name (must match key in config/bots_config.json)")
    args = parser.parse_args()
    
    logger.info(f"Initializing bot: {args.bot}")
    bot = TelegramBot(args.bot)
    
    if not bot.initialize():
        logger.critical(f"Failed to initialize bot '{args.bot}'")
        return 1
    
    logger.info("Bot initialization successful, starting...")
    
    # Change working directory to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    logger.info(f"Changed working directory to project root: {project_root}")
    
    try:
        bot.run()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.critical(f"Error: {e}", exc_info=True)
    finally:
        logger.info("Bot shutting down...")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
