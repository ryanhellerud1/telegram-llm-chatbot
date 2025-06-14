from telegram import Update
from telegram.ext import ContextTypes
from typing import Any

async def handle_start_command(bot: Any, update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id if update.effective_chat else None
    if chat_id:
        welcome_message = bot.config.welcome_message_template.format(
            member_full_name=update.effective_user.full_name if update.effective_user else "there"
        )
        await context.bot.send_message(chat_id=chat_id, text=welcome_message)
    bot.logger.info(f"[ChatID: {chat_id}] Sent welcome message for /start command.")

async def handle_ask_command(bot: Any, update: Update, context: ContextTypes.DEFAULT_TYPE):
    # TODO: Implement /ask command logic here or call a method if you add it to TelegramBot
    pass

async def handle_draw_command(bot: Any, update: Update, context: ContextTypes.DEFAULT_TYPE):
    # TODO: Implement /draw command logic here or call a method if you add it to TelegramBot
    pass

async def handle_new_members(bot: Any, update: Update, context: ContextTypes.DEFAULT_TYPE):
    # TODO: Implement new member welcome logic here or call a method if you add it to TelegramBot
    pass

async def handle_text_messages(bot: Any, update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id if update.effective_chat else None
    user_id = update.effective_user.id if update.effective_user else None
    username = update.effective_user.username if update.effective_user else "Unknown"
    text = update.message.text if update.message else ""
    bot.logger.info(f"[ChatID: {chat_id}] Received text message from {username}: {text}")
    if chat_id and user_id:
        bot.add_message_to_history(chat_id, "user", text, user_id, username)
    # Optionally, you can trigger a reply or further processing here
