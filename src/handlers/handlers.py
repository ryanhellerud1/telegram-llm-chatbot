from telegram import Update
from telegram.ext import ContextTypes
from src.bot.telegram_bot import TelegramBot

async def handle_start_command(bot: TelegramBot, update: Update, context: ContextTypes.DEFAULT_TYPE):
    await bot.handle_start_command(update, context)

async def handle_ask_command(bot: TelegramBot, update: Update, context: ContextTypes.DEFAULT_TYPE):
    await bot.handle_ask_command(update, context)

async def handle_draw_command(bot: TelegramBot, update: Update, context: ContextTypes.DEFAULT_TYPE):
    await bot.handle_draw_command(update, context)

async def handle_new_members(bot: TelegramBot, update: Update, context: ContextTypes.DEFAULT_TYPE):
    await bot.handle_new_members(update, context)

async def handle_text_messages(bot: TelegramBot, update: Update, context: ContextTypes.DEFAULT_TYPE):
    await bot.handle_text_messages(update, context)
