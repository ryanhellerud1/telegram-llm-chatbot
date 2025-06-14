import logging
import os
from telegram import Update, error as telegram_error
from telegram.ext import ContextTypes
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.bot.telegram_bot import TelegramBot

# It's better to use a module-level logger for utility files
logger = logging.getLogger(__name__)

async def handle_telegram_error(
    bot_instance: 'TelegramBot',
    update: Optional[Update], # Changed from Optional[object] to Optional[Update]
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """
    Global error handler for the Telegram bot application.

    This function is called by the `telegram.ext.Application` when an error occurs
    that is not handled by other error handlers. It logs the error and handles
    specific cases like `telegram.error.Conflict`.

    Args:
        bot_instance: The instance of the `TelegramBot` class that this error pertains to.
                      This allows access to bot-specific configurations and states,
                      such as its logger or application object.
        update: The `telegram.Update` object that caused the error, if available.
                It might be None for some types of errors (e.g., job queue errors).
        context: The `telegram.ext.ContextTypes.DEFAULT_TYPE` object associated with
                 the error, containing the `error` attribute itself.
    """
    error = context.error # Shortcut to the error

    if isinstance(error, telegram_error.Conflict):
        # This error means another instance of the bot with the same token is running.
        # The current instance should terminate to avoid conflicts.
        logger.critical( # Changed to critical as this is a fatal error for this instance
            f"Conflict error for bot '{bot_instance.bot_name}': Another instance is running. "
            "This instance will terminate."
        )
        if bot_instance.application and bot_instance.application.running:
            try:
                # Attempt to stop polling, though it might not always succeed if the conflict
                # has already severely impacted the bot's connection.
                bot_instance.application.stop_running()
                logger.info(f"Application stop_running() called for bot '{bot_instance.bot_name}'.")
            except Exception as e_stop:
                logger.error(f"Error while trying to stop application for bot '{bot_instance.bot_name}' during conflict: {e_stop}", exc_info=True)

        # os._exit(1) is a hard exit. It's used here because a Conflict error
        # usually means this instance cannot recover and should not continue.
        # This bypasses finally clauses and other cleanup, which might be acceptable
        # in this specific scenario to prevent two instances from interfering.
        os._exit(1) # Force exit the current process

    # For all other types of errors:
    error_message = f"Unhandled exception for bot '{bot_instance.bot_name}': {error}"

    # Enhance error message with details from the update, if available
    if update:
        if update.effective_chat:
            error_message += f" | Chat ID: {update.effective_chat.id}"
            if update.effective_chat.title:
                 error_message += f" (Title: {update.effective_chat.title})"
        if update.effective_user:
            error_message += f" | User ID: {update.effective_user.id}"
            if update.effective_user.username:
                 error_message += f" (@{update.effective_user.username})"
        if update.message and update.message.text:
            error_message += f" | Message: \"{update.message.text[:100]}\"" # Log first 100 chars

    # Log the error with full traceback using exc_info=error
    # Using bot_instance.logger to ensure the log is associated with the specific bot.
    bot_instance.logger.error(error_message, exc_info=error)

    # TODO: Consider sending a generic error message to the user or admin for certain non-critical errors.
    # Example (be careful about error loops or spamming users):
    # if update and update.effective_chat and not isinstance(error, (telegram_error.NetworkError, telegram_error.TimedOut)):
    #     try:
    #         await context.bot.send_message(
    #             chat_id=update.effective_chat.id,
    #             text="Oops! Something went wrong. The admin has been notified. Please try again later."
    #         )
    #     except Exception as send_error:
    #         bot_instance.logger.error(
    #             f"Failed to send error notification to user in chat {update.effective_chat.id}: {send_error}",
    #             exc_info=send_error
    #         )
