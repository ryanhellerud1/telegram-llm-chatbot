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
from src.config.config_manager import ConfigManager
from src.config import constants as const
from src.handlers.handlers import handle_start_command, handle_new_members, handle_text_messages
from src.handlers.handlers import handle_ask_command, handle_draw_command
from src.handlers.error_handler import handle_telegram_error
from src.utils.helpers import get_username, is_user_replying_to_user
from src.utils.logging_utils import setup_bot_logging
from src.services.image_client import ImageGenerationClient # Added import


# Removed duplicate BotConfig definition


class TelegramBot:
    """
    A class-based Telegram bot with LLM integration and periodic responses.
    """
    
    _running = False  # Class variable to track running state
    _instances = weakref.WeakSet()  # type: weakref.WeakSet[TelegramBot]
    _shutdown_event = asyncio.Event() # Class level event to signal shutdown

    @staticmethod
    async def _handle_exit_signal_async(sig: int, frame: Optional[Any]) -> None:
        """
        Asynchronous signal handler for graceful shutdown.

        This method is registered to handle signals like SIGINT and SIGTERM.
        It sets a global shutdown event and attempts to synchronously trigger
        the shutdown process for all active bot instances.

        Args:
            sig: The signal number.
            frame: The current stack frame.
        """
        # Use a generic logger if bot-specific one isn't available in this static context easily
        static_logger = logging.getLogger("TelegramBot.SignalHandler")
        static_logger.info(f"Signal {signal.Signals(sig).name} received. Initiating shutdown sequence...")
        
        TelegramBot._shutdown_event.set() # Signal all instances/parts of the application to shutdown

        # Synchronously iterate and call shutdown on instances.
        # This is a simplification. A fully async shutdown might involve gathering tasks.
        active_instances = list(TelegramBot._instances)
        static_logger.info(f"Found {len(active_instances)} active bot instance(s) to shut down.")
        for bot_instance in active_instances:
            if hasattr(bot_instance, 'shutdown_sync') and callable(bot_instance.shutdown_sync):
                try:
                    static_logger.info(f"Calling shutdown_sync() for bot '{bot_instance.bot_name}'...")
                    bot_instance.shutdown_sync()
                except Exception as e:
                    static_logger.error(f"Error during shutdown_sync for bot '{bot_instance.bot_name}': {e}", exc_info=True)
            else:
                static_logger.warning(f"Bot instance '{getattr(bot_instance, 'bot_name', 'Unknown')}' lacks a callable shutdown_sync method.")
        
        # Consider if a more forceful exit is needed after a timeout if graceful shutdown hangs.
        # For now, relying on application.stop_running() to allow main loop to exit.

    def __init__(self, bot_name: str):
        """
        Initializes a TelegramBot instance.

        Args:
            bot_name: The unique name for this bot instance, used for configuration and logging.
        """
        self.bot_name: str = bot_name
        self.config_manager: Optional[ConfigManager] = None
        self.application: Optional[Application] = None
        # Logger will be properly initialized in the `initialize` method.
        # For now, get a basic logger to capture early messages if any.
        self.logger: logging.Logger = logging.getLogger(f"bot.{bot_name}") # Updated logger name
        self._ssl_context: ssl.SSLContext = ssl.create_default_context(cafile=certifi.where())

        TelegramBot._instances.add(self)
        self.llm_client: Optional[LLMClient] = None
        self.image_client: Optional[ImageGenerationClient] = None
        self.chat_history_manager: ChatHistoryManager = ChatHistoryManager(history_max_length=const.HISTORY_MAX_LENGTH)
        self.periodic_jobs: Dict[int, Any] = {} # Store Job instances
    
    def ensure_chat_data(self, chat_id: int) -> ChatData:
        """
        Ensures that a `ChatData` object exists for the given chat ID and returns it.
        If one does not exist, it is created by the `ChatHistoryManager`.

        Args:
            chat_id: The ID of the chat.

        Returns:
            The `ChatData` object for the specified chat.
        """
        return self.chat_history_manager.ensure_chat_data(chat_id)
    
    def add_message_to_history(self, chat_id: int, role: str, content: str, 
                             user_id: int, username: str = "Unknown", 
                             is_reply_to_user: bool = False) -> None:
        """
        Adds a message to the chat history for a given chat ID.

        Args:
            chat_id: The ID of the chat.
            role: The role of the message sender (e.g., "user", "assistant").
            content: The textual content of the message.
            user_id: The ID of the user who sent the message.
            username: The username of the message sender. Defaults to "Unknown".
            is_reply_to_user: Boolean indicating if this message is a reply from the bot to a user.
        """
        self.chat_history_manager.add_message_to_history(
            chat_id, role, content, user_id, username, is_reply_to_user, logger=self.logger
        )
        # Logging message content can be verbose and sensitive. Consider if it's needed.
        self.logger.info(
            f"[ChatID: {chat_id}] Message (role: {role}, user: {username}/{user_id}, reply_to_user: {is_reply_to_user}) "
            f"added. History length: {len(self.chat_history_manager.get_chat_data(chat_id).message_history)}"
        )
    
    async def periodic_llm_callback(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Callback function for periodic LLM responses triggered by the job queue.

        This method fetches chat data, prepares messages for the LLM,
        gets a response, and sends it back to the chat.

        Args:
            context: The `telegram.ext.ContextTypes.DEFAULT_TYPE` provided by the job queue.
                     It contains job data, including `chat_id`.
        """
        if not context.job or not isinstance(context.job.data, dict):
            self.logger.error("Periodic LLM callback called with invalid job context or data.")
            return

        chat_id: int = context.job.data.get('chat_id')
        if not chat_id:
            self.logger.error("Periodic LLM callback: chat_id not found in job data.")
            return

        self.logger.info(f"[ChatID: {chat_id}] Periodic LLM callback triggered (interval tick).")
        
        self.logger.info(f"[ChatID: {chat_id}] Attempting periodic LLM response...")
        chat_data = self.chat_history_manager.get_chat_data(chat_id)

        if not chat_data or not chat_data.new_activity_flag:
            self.logger.info(f"[ChatID: {chat_id}] No new activity; skipping periodic LLM call.")
            return
        
        if not self.llm_client:
            self.logger.error(f"[ChatID: {chat_id}] LLMClient not initialized. Cannot make periodic LLM call.")
            return

        messages = self.llm_client.prepare_messages_for_llm(
            chat_id,
            self.chat_history_manager,
            include_introduction=True # For periodic jobs, an intro might be good
        )

        if not messages:
            self.logger.info(f"[ChatID: {chat_id}] No messages prepared for LLM (history might be empty or only system prompt). Skipping.")
            return

        response_text = await self.llm_client.get_llm_response(chat_id, messages, for_periodic_job=True)
        
        if response_text and not any(response_text.startswith(prefix) for prefix in ["*error", "*system", "*critical"]):
            # Tag the last active user if appropriate
            last_username = "Unknown"
            if chat_data.message_history: # Check if history is not empty
                for msg in reversed(chat_data.message_history):
                    if msg.get("role") == "user" and msg.get("username"):
                        last_username = msg["username"]
                        break
            if last_username != "Unknown" and not response_text.startswith(f"@{last_username}"):
                response_text = f"@{last_username} {response_text}"
            
            self.logger.info(f"[ChatID: {chat_id}] Sending periodic LLM response (length: {len(response_text)}). Preview: '{response_text[:100]}...'")
            try:
                await context.bot.send_message(chat_id=chat_id, text=response_text)
                self.add_message_to_history(
                    chat_id=chat_id,
                    role="assistant",
                    content=response_text,
                    user_id=context.bot.id, # Bot's own ID
                    username=self.config_manager.get('bot_username_internal', 'Bot')
                )
                chat_data.new_activity_flag = False
                self.logger.info(f"[ChatID: {chat_id}] Successfully sent periodic response and reset activity flag.")
            except telegram_error.TelegramError as e:
                self.logger.error(f"[ChatID: {chat_id}] TelegramError sending periodic message: {e}", exc_info=True)
                # Potentially retry or handle specific Telegram errors (e.g., chat not found, bot blocked)
            except Exception as e: # Catch other unexpected errors during send or history add
                self.logger.error(f"[ChatID: {chat_id}] Unexpected error sending periodic message or updating history: {e}", exc_info=True)
        elif response_text: # LLM returned a message prefixed with an error indicator
            self.logger.warning(f"[ChatID: {chat_id}] Periodic LLM call returned a non-fatal error/info message: {response_text}")
        else: # No response_text from LLM
            self.logger.error(f"[ChatID: {chat_id}] Periodic LLM call returned no response or an empty response.")
    
    def setup_handlers(self) -> None:
        """
        Sets up the command and message handlers for the bot.
        This includes a global error handler.
        Raises RuntimeError if the application is not initialized.
        """
        if not self.application:
            raise RuntimeError("TelegramBot application not initialized before setting up handlers.")
        
        # Register the global error handler
        self.application.add_error_handler(
            lambda update, context: handle_telegram_error(self, update, context)
        )

        # Define handlers
        # Using lambdas to pass `self` (the bot instance) to handler functions
        command_handlers = [
            CommandHandler("start", lambda u, c: handle_start_command(self, u, c)),
            CommandHandler("ask", lambda u, c: handle_ask_command(self, u, c)),
            CommandHandler("draw", lambda u, c: handle_draw_command(self, u, c)),
        ]
        message_handlers = [
            MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, lambda u, c: handle_new_members(self, u, c)),
            MessageHandler(filters.TEXT & ~filters.COMMAND, lambda u, c: handle_text_messages(self, u, c)),
        ]

        all_handlers = command_handlers + message_handlers
        for handler in all_handlers:
            self.application.add_handler(handler)

        self.logger.info(f"Registered {len(all_handlers)} handlers for bot '{self.bot_name}'.")
    
    def initialize(self) -> bool:
        """
        Initializes the bot: loads configuration, sets up logging,
        initializes API clients (LLM, Image), and sets up Telegram application handlers.

        Returns:
            True if initialization was successful, False otherwise.
        """
        self.logger.info(f"Initializing bot '{self.bot_name}'...")
        # Determine project root assuming this file is in src/bot/
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Construct path to the bot's specific JSON config file
        # Example: config/bots/MyBotName_config.json
        config_file_name = f"{self.bot_name}_config.json"
        config_file_path = os.path.join(project_root, 'config', 'bots', config_file_name)
        
        # Optional: Construct path to a .env file, perhaps in project_root or config dir
        dotenv_path = os.path.join(project_root, '.env') # Standard .env location

        self.config_manager = ConfigManager(config_path=config_file_path, dotenv_path=dotenv_path)
        
        # Setup instance-specific logger using ConfigManager values
        env_prefix = self.config_manager.get('env_prefix', self.bot_name.upper()) # Default prefix if not in config
        log_level_key = f"{env_prefix}_LOG_LEVEL"
        # Use bot_name_display from config for logger, fallback to self.bot_name
        logger_display_name = self.config_manager.get('bot_name_display', self.bot_name)

        self.logger = setup_bot_logging(
            logger_name=f"bot.{self.bot_name}", # More specific logger name
            log_level_env_key=log_level_key,
            bot_display_name_or_env_key=logger_display_name
        )
        
        # Validate essential configurations
        if not self.config_manager.get('telegram_bot_token'):
            self.logger.critical("TELEGRAM_BOT_TOKEN is missing in configuration.")
            return False
        if not self.config_manager.get('system_prompt'): # Checks if loaded from file or ENV
            self.logger.warning("System prompt is not configured. Using default.") # Changed to warning as default exists
            # No need to return False, as a default is provided by ConfigManager.get or prepare_messages_for_llm
        if not self.config_manager.get('openrouter_model'): # Assuming this is for the primary LLM
            self.logger.critical("OPENROUTER_MODEL (for LLM) is missing in configuration.")
            return False
        
        # Initialize LLMClient
        # Ensure all required keys are present in config_manager or have defaults
        self.llm_client = LLMClient(
            api_key=self.config_manager.get('openrouter_api_key'), # This might be general or OpenRouter specific
            openrouter_api_key=self.config_manager.get('openrouter_api_key'), # Explicit OpenRouter key
            model=self.config_manager.get('openrouter_model'),
            bot_username_internal=self.config_manager.get('bot_username_internal', 'UnknownBot'),
            bot_name_display=self.config_manager.get('bot_name_display', self.bot_name),
            logger=self.logger,
            ssl_context=self._ssl_context,
            config_manager=self.config_manager
        )

        # Initialize ImageGenerationClient
        # No specific API key needed in constructor if passed per-call
        self.image_client = ImageGenerationClient(
            logger=self.logger,
            ssl_context=self._ssl_context
            # Image provider name uses default from const.IMAGE_PROVIDER_NAME
            # If different image providers need different keys in constructor, adjust here.
        )
        
        # Initialize Telegram PTB Application
        self.application = Application.builder().token(self.config_manager.get('telegram_bot_token')).build()
        self.setup_handlers() # Register all handlers

        self.logger.info(f"Bot '{self.config_manager.get('bot_name_display', self.bot_name)}' initialized successfully.")
        return True
    
    async def _check_telegram_api(self) -> bool:
        """
        Performs a quick check to see if the Telegram Bot API is reachable
        with the configured bot token. Helps detect token issues early.

        Returns:
            True if the API call to `getMe` is successful, False otherwise.
        """
        if not self.config_manager or not self.config_manager.get('telegram_bot_token'):
            self.logger.error("Cannot check Telegram API: Bot token not configured.")
            return False
        try:
            connector = aiohttp.TCPConnector(ssl=self._ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                # Using a short timeout for this check
                timeout = aiohttp.ClientTimeout(total=const.REQUEST_TIMEOUT / 2) # Use half of general request timeout
                async with session.get(
                    f"https://api.telegram.org/bot{self.config_manager.get('telegram_bot_token')}/getMe",
                    timeout=timeout
                ) as response:
                    if response.status == 409: # Conflict, another instance is running
                        self.logger.error("Telegram API check: Conflict (409). Another instance might be running.")
                        return False # Treat conflict as a failure for this check's purpose
                    response.raise_for_status() # Raise HTTPError for bad responses (4XX or 5XX)
                    self.logger.info("Telegram API check successful (getMe).")
                    return True
        except aiohttp.ClientConnectorError as e:
            self.logger.error(f"Telegram API check failed: Connection error - {e}")
            return False
        except aiohttp.ClientResponseError as e:
            self.logger.error(f"Telegram API check failed: HTTP {e.status} - {e.message}")
            return False
        except asyncio.TimeoutError:
            self.logger.warning("Telegram API check failed: Request timed out.")
            return False
        except Exception as e: # Catch any other unexpected errors
            self.logger.error(f"Telegram API check failed with an unexpected error: {e}", exc_info=True)
            return False

    def run(self) -> None:
        """
        Runs the bot's main polling loop.
        This method is blocking and will continue until the bot is stopped
        (e.g., by a signal or if `stop_running()` is called on the application).
        """
        if not self.application:
            self.logger.critical(f"Bot '{self.bot_name}' cannot run: Application not initialized.")
            return
        if not self.config_manager:
            self.logger.critical(f"Bot '{self.bot_name}' cannot run: ConfigManager not initialized.")
            return

        bot_display_name = self.config_manager.get('bot_name_display', self.bot_name)
        self.logger.info(f"Starting bot polling for '{bot_display_name}'...")

        # The run_polling method is blocking. It will exit when application.stop_running() is called.
        try:
            self.application.run_polling(
                allowed_updates=Update.ALL_TYPES, # Consider specifying only needed updates
                drop_pending_updates=True,        # Good for ensuring a clean state on startup
                close_loop=False                  # Important for PTB v20+ if managing loop externally
            )
        except telegram_error.TelegramError as te:
            # Specific Telegram errors during polling (e.g., network issues if not caught by getMe)
            self.logger.critical(f"TelegramError during polling for '{bot_display_name}': {te}", exc_info=True)
        except Exception as e:
            # Catch-all for any other unexpected error during polling
            self.logger.critical(f"Unexpected error during polling for '{bot_display_name}': {e}", exc_info=True)
        finally:
            # This block executes when run_polling() stops, regardless of the reason.
            self.logger.info(f"Bot polling stopped for '{bot_display_name}'.")
            # Instance removal from _instances is handled by the signal handler or main cleanup.
            # If an explicit .shutdown() async method were added to TelegramBot,
            # it could be called from main_async's finally block.
            # For now, shutdown_sync is called by the signal handler.
            # The application.shutdown() is called in main_async's finally block.


    def shutdown_sync(self) -> None:
        """
        Performs synchronous shutdown actions for the bot instance.
        This includes stopping the Telegram application's updater.
        Called by the signal handler.
        """
        self.logger.info(f"Bot '{self.bot_name}': shutdown_sync() called.")
        if not self.application:
            self.logger.warning(f"Bot '{self.bot_name}': No application to shut down during shutdown_sync.")
            return

        if self.application.running:
            self.logger.info(f"Bot '{self.bot_name}': Application is running, calling stop_running().")
            try:
                self.application.stop_running()
                self.logger.info(f"Bot '{self.bot_name}': stop_running() successfully called.")
            except Exception as e:
                self.logger.error(f"Bot '{self.bot_name}': Error calling stop_running(): {e}", exc_info=True)
        else:
            self.logger.info(f"Bot '{self.bot_name}': Application was not running when shutdown_sync was called.")

        # Other synchronous cleanup tasks for the bot instance can be added here.
        # For example, closing database connections if they were managed synchronously.
        self.logger.info(f"Bot '{self.bot_name}': Synchronous shutdown actions completed.")


async def main_async() -> int:
    """
    Asynchronous main entry point for the bot application.

    Initializes and runs a single bot instance based on command-line arguments.
    Handles overall application setup and graceful shutdown coordination.

    Returns:
        An exit code (0 for success, 1 for failure).
    """
    # Setup basic logging for the main process before bot-specific loggers take over.
    # This ensures visibility during early startup or if bot initialization fails.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    main_process_logger = logging.getLogger("MainProcess") # Use a distinct name
    
    parser = argparse.ArgumentParser(description="Run a Telegram Bot with a specific personality/configuration.")
    parser.add_argument(
        "--bot",
        required=True,
        help="Bot name (must correspond to a configuration file, e.g., 'MyBot' for 'MyBot_config.json')."
    )
    args = parser.parse_args()
    
    main_process_logger.info(f"Attempting to initialize and run bot: '{args.bot}'")
    
    # Instantiate the bot. Logging within the bot will use its specific configured logger.
    bot = TelegramBot(args.bot)
    
    if not bot.initialize(): # This also sets up the bot's own logger.
        main_process_logger.critical(f"Failed to initialize bot '{args.bot}'. Exiting.")
        return 1 # Indicate failure

    # The `initialize` method has already set the project root as CWD if needed.
    # No need to change os.chdir here unless specifically required for other reasons.

    main_process_logger.info(f"Bot '{args.bot}' initialized. Starting polling...")
    
    try:
        # bot.run() is blocking and will only return when polling stops.
        # The signal handler (TelegramBot._handle_exit_signal_async) calls application.stop_running(),
        # which causes run_polling() to eventually exit.
        await asyncio.shield(bot.run()) # Shield to prevent KeyboardInterrupt from directly stopping run
    except asyncio.CancelledError:
        bot.logger.info(f"Bot '{args.bot}' run task was cancelled (likely during shutdown).")
    except Exception as e:
        # Log with the bot's own logger if available, otherwise main process logger.
        logger_to_use = bot.logger if bot.logger else main_process_logger
        logger_to_use.critical(f"Bot '{args.bot}' failed unexpectedly during run: {e}", exc_info=True)
        return 1 # Indicate failure
    finally:
        # This block executes after bot.run() completes or is cancelled.
        logger_to_use = bot.logger if bot.logger else main_process_logger
        logger_to_use.info(f"Bot '{args.bot}' has stopped. Main async task is concluding.")

        # Perform final asynchronous cleanup for the PTB application.
        if bot.application and hasattr(bot.application, 'shutdown') and asyncio.iscoroutinefunction(bot.application.shutdown):
            logger_to_use.info(f"Performing async shutdown of PTB application for bot '{args.bot}'...")
            try:
                await bot.application.shutdown()
                logger_to_use.info(f"PTB application for bot '{args.bot}' shutdown complete.")
            except Exception as e_shutdown:
                logger_to_use.error(f"Error during PTB application shutdown for bot '{args.bot}': {e_shutdown}", exc_info=True)

        # Additional async cleanup for the bot instance itself could be awaited here if defined.
        # e.g., if bot.async_cleanup() existed: await bot.async_cleanup()

        # Remove instance from global tracking set as it's now fully stopped.
        if bot in TelegramBot._instances:
            TelegramBot._instances.remove(bot)
            logger_to_use.info(f"Bot instance '{args.bot}' removed from global tracking set.")

    return 0 # Indicate success


def main() -> int:
    """
    Synchronous main entry point that sets up signal handling and runs the
    asynchronous `main_async` function.
    """
    # Get the current event loop.
    # Python 3.7+ `asyncio.get_running_loop()` is preferred if inside a coroutine,
    # but `get_event_loop()` is fine here at the top level of a sync function.
    loop = asyncio.get_event_loop()

    # Register signal handlers for SIGINT (Ctrl+C) and SIGTERM.
    # The lambda creates an asyncio task to run the async signal handler.
    # This is a common pattern for integrating synchronous signal handling with asyncio.
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig,
            lambda s=sig: asyncio.create_task(TelegramBot._handle_exit_signal_async(s, None))
        )
        # Note: `signal.signal` is an alternative but `loop.add_signal_handler` is
        # generally preferred for asyncio applications for better integration.
        # If using `signal.signal`, it would be:
        # signal.signal(sig, lambda s, f: asyncio.create_task(TelegramBot._handle_exit_signal_async(s, f)))

    exit_code: int = 0
    try:
        # Run the main asynchronous logic until it completes.
        exit_code = loop.run_until_complete(main_async())
    except KeyboardInterrupt:
        # This might still occur if the initial signal registration or loop setup is interrupted.
        # Or if the signal handler itself doesn't stop the loop properly.
        logging.getLogger("MainProcess").info("KeyboardInterrupt caught in main synchronous wrapper. Forcing exit.")
        exit_code = 1 # Non-zero exit code for interruption
    except Exception as e:
        # Catch any other unexpected errors during the execution of main_async or loop management.
        logging.getLogger("MainProcess").critical(f"Critical error in main synchronous execution: {e}", exc_info=True)
        exit_code = 1 # Non-zero exit code for critical failure
    finally:
        # Important: Clean up the event loop.
        # Gather any remaining tasks and cancel them.
        # Based on Python docs for asyncio cleanup.
        pending_tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
        if pending_tasks:
            logging.getLogger("MainProcess").info(f"Cancelling {len(pending_tasks)} outstanding tasks...")
            for task in pending_tasks:
                task.cancel()
            # Allow tasks to process their cancellation
            loop.run_until_complete(asyncio.gather(*pending_tasks, return_exceptions=True))
            logging.getLogger("MainProcess").info("Outstanding tasks cancelled.")

        # It's generally recommended to close the loop only when the application is completely finished.
        # If other parts of your system might still need the loop, be cautious.
        # For a standalone script like this, it's usually safe.
        # However, `run_until_complete` might handle some of this.
        # If issues arise, this is a point to review.
        # loop.close()
        # logging.getLogger("MainProcess").info("Asyncio event loop closed.")
        pass # Often loop.close() is not needed or can cause issues if not handled carefully

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
