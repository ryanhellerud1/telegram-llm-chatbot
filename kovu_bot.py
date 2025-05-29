# Narlyai_bot.py
# This is the main file for the Narly ai Telegram bot.

import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Define your bot token here
# IMPORTANT: Replace "YOUR_TELEGRAM_BOT_TOKEN" with your actual bot token
BOT_TOKEN = "8002927976:AAENzGrmogB9DJX5xpmz8li_tan6Ii1LfOo"



async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message when the /start command is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}! I am Narly ai, your friendly Telegram bot.",
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a help message when the /help command is issued."""
    await update.message.reply_text("Available commands:\n/start - Start the bot\n/help - Show this help message")


async def welcome_new_members(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Welcomes new members to the chat."""
    if update.message and update.message.new_chat_members:
        for new_member in update.message.new_chat_members:
            if new_member.is_bot:
                continue  # Don't welcome other bots
            await update.message.reply_text(
                f"Welcome to the Narly ai group, {new_member.mention_html()}! Glad to have you here. I am the AI overlord here."
            )


def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(BOT_TOKEN).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # Handler for new members joining the chat
    application.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, welcome_new_members))

    # Run the bot until the user presses Ctrl-C
    logger.info("Starting bot...")
    application.run_polling()
    logger.info("Bot stopped.")

if __name__ == "__main__":
    main()
