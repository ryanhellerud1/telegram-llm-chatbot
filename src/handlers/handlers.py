from telegram import Update
from telegram.ext import ContextTypes, JobQueue # Added JobQueue for type hint if needed by context
from typing import TYPE_CHECKING, Optional # Added TYPE_CHECKING and Optional

if TYPE_CHECKING:
    from src.bot.telegram_bot import TelegramBot # Forward reference for type hinting

async def handle_start_command(bot: 'TelegramBot', update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles the /start command. Sends a welcome message to the user.
    """
    chat_id: Optional[int] = update.effective_chat.id if update.effective_chat else None
    user_full_name: str = update.effective_user.full_name if update.effective_user else "there"

    if chat_id:
        # Ensure config_manager and welcome_message_template are available
        if bot.config_manager and bot.config_manager.get('welcome_message_template'):
            welcome_message_template = bot.config_manager.get('welcome_message_template', "Welcome {member_full_name}!")
            welcome_message = welcome_message_template.format(member_full_name=user_full_name)
        else:
            # Fallback if config or template is missing
            welcome_message = f"Welcome {user_full_name}!"
            bot.logger.warning(f"[ChatID: {chat_id}] Welcome message template not found in config. Using default.")

        await context.bot.send_message(chat_id=chat_id, text=welcome_message)
        bot.logger.info(f"[ChatID: {chat_id}] Sent welcome message to '{user_full_name}' for /start command.")
    else:
        bot.logger.warning("Could not determine chat_id for /start command.")

async def handle_ask_command(bot: 'TelegramBot', update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Placeholder for handling the /ask command.
    TODO: Implement LLM interaction for direct questions.
    """
    bot.logger.info(f"TODO: /ask command invoked by {update.effective_user.username if update.effective_user else 'Unknown'}")
    await update.message.reply_text("The /ask command is not yet implemented. Please try asking your question directly.")
    # TODO: Implement /ask command logic here or call a method if you add it to TelegramBot

async def handle_draw_command(bot: 'TelegramBot', update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles the /draw command for image generation.
    Extracts a prompt, calls the ImageGenerationClient, and sends the resulting image.
    """
    if not update.effective_chat or not update.effective_user or not update.message:
        bot.logger.warning("/draw command received without effective_chat, user or message.")
        return

    chat_id: int = update.effective_chat.id
    user_id: int = update.effective_user.id
    username: str = update.effective_user.username or f"UserID_{user_id}" # Fallback username

    prompt: str = ""
    if context.args:
        prompt = " ".join(context.args)
    elif update.message.reply_to_message and update.message.reply_to_message.text:
        prompt = update.message.reply_to_message.text
    else:
        await update.message.reply_text(
            "Please provide a prompt for the image. Usage: /draw <your prompt> "
            "or reply to a message containing the prompt with /draw."
        )
        return

    if not prompt.strip(): # Check if prompt is empty or whitespace
        await update.message.reply_text("Prompt is empty. Please provide some text to generate an image.")
        return

    bot.logger.info(f"[ChatID: {chat_id}] User {username} (ID: {user_id}) initiated /draw with prompt: '{prompt[:100]}...'")

    if not bot.config_manager or not bot.image_client:
        bot.logger.error(f"[ChatID: {chat_id}] Bot is not properly initialized (missing ConfigManager or ImageClient). Cannot generate image.")
        await update.message.reply_text("Sorry, the image generation service is not properly configured on my end.")
        return

    # Retrieve API key for image generation. Key name in config can be specific.
    image_api_key: Optional[str] = bot.config_manager.get('modelslab_api_key') # Default key name
    if not image_api_key: # Fallback or alternative key names
        image_api_key = bot.config_manager.get('image_generation_api_key')

    if not image_api_key:
        bot.logger.error(f"[ChatID: {chat_id}] Image generation API key (e.g., 'modelslab_api_key') not found in configuration.")
        await update.message.reply_text("Sorry, the image generation service API key is not configured. Please contact the admin.")
        return

    await update.message.reply_text("🎨 Your image is being conjured from the digital ether... please wait a moment!")

    try:
        image_url = await bot.image_client.generate_image(prompt=prompt, api_key=image_api_key)

        if image_url:
            bot.logger.info(f"[ChatID: {chat_id}] Successfully generated image for prompt '{prompt[:50]}...'. URL: {image_url}")
            await context.bot.send_photo(
                chat_id=chat_id,
                photo=image_url,
                caption=f"🖼️ Here's your masterpiece for: \"{prompt[:150]}\"" # Max caption length for Telegram is 1024
            )
        else:
            bot.logger.error(f"[ChatID: {chat_id}] Failed to generate image for prompt '{prompt[:50]}...'. No URL returned from client.")
            await update.message.reply_text("Sorry, I couldn't generate an image for that prompt. The digital muses were uncooperative. Try a different prompt or try again later.")

    except Exception as e:
        bot.logger.error(f"[ChatID: {chat_id}] Exception during /draw command for prompt '{prompt[:50]}...': {e}", exc_info=True)
        await update.message.reply_text("An unexpected error occurred while trying to generate the image. The tech-sorcerers have been alerted!")


async def handle_new_members(bot: 'TelegramBot', update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles new members joining a chat.
    TODO: Implement logic to welcome new members, perhaps with a custom message.
    """
    if not update.message or not update.message.new_chat_members:
        return # Should not happen for NEW_CHAT_MEMBERS update

    for member in update.message.new_chat_members:
        chat_id: int = update.effective_chat.id
        username: str = member.username or member.full_name
        bot.logger.info(f"[ChatID: {chat_id}] New member joined: {username} (ID: {member.id})")
        # TODO: Send a welcome message, potentially customized from bot.config_manager
        # Example: await update.message.reply_text(f"Welcome {member.mention_html()} to the group!")


async def handle_text_messages(bot: 'TelegramBot', update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles regular text messages (not commands).
    Adds the message to chat history and schedules a periodic LLM job if not already running for the chat.
    """
    if not update.message or not update.message.text or not update.effective_chat or not update.effective_user:
        bot.logger.debug("Text message handler received an update without message text, chat, or user.")
        return # Ignore messages without text or essential context

    chat_id: int = update.effective_chat.id
    user_id: int = update.effective_user.id
    username: str = update.effective_user.username or f"UserID_{user_id}" # Fallback username
    text: str = update.message.text

    bot.logger.info(f"[ChatID: {chat_id}] Received text message from {username} (ID: {user_id}): \"{text[:100]}\"...")

    bot.add_message_to_history(chat_id, "user", text, user_id, username)

    # Ensure job_queue is available on context
    if not context.job_queue:
        bot.logger.error(f"[ChatID: {chat_id}] JobQueue not available in context. Cannot schedule periodic LLM job.")
        return

    # Schedule periodic job if not already scheduled for this chat
    # Ensure periodic_jobs attribute exists and is a dict
    if not hasattr(bot, 'periodic_jobs') or not isinstance(bot.periodic_jobs, dict):
        bot.logger.error(f"[ChatID: {chat_id}] bot.periodic_jobs not found or not a dict. Cannot schedule job.")
        bot.periodic_jobs = {} # Initialize if missing, though it should be in __init__

    if chat_id not in bot.periodic_jobs:
        bot.logger.info(f"[ChatID: {chat_id}] Scheduling new periodic LLM job.")
        # Ensure PERIODIC_JOB_INTERVAL_SECONDS and PERIODIC_JOB_FIRST_RUN_DELAY_SECONDS are accessible
        # Typically these would be constants or part of config. Assuming they are on `bot` or `bot.config_manager`
        # For this example, assuming they are available via bot.config_manager or constants imported in bot module
        interval_seconds = bot.config_manager.get('periodic_job_interval_seconds', const.PERIODIC_JOB_INTERVAL_SECONDS)
        first_run_delay = bot.config_manager.get('periodic_job_first_run_delay', const.PERIODIC_JOB_FIRST_RUN_DELAY_SECONDS)

        job = context.job_queue.run_repeating(
            bot.periodic_llm_callback,
            interval=interval_seconds,
            first=first_run_delay,
            data={"chat_id": chat_id}, # Pass chat_id in job data
            name=f"periodic_llm_{chat_id}"
        )
        bot.periodic_jobs[chat_id] = job
        bot.logger.info(f"[ChatID: {chat_id}] Scheduled periodic LLM job with interval {interval_seconds}s.")
    else:
        bot.logger.debug(f"[ChatID: {chat_id}] Periodic LLM job already exists.")

    # Optionally, set new_activity_flag if that's part of the logic to trigger LLM immediately
    chat_data = bot.chat_history_manager.get_chat_data(chat_id)
    if chat_data:
        chat_data.new_activity_flag = True
