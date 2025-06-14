def get_username(user):
    """Extract username from user object."""
    return (
        user.username or 
        getattr(user, 'first_name', None) or 
        getattr(user, 'full_name', None) or 
        "Unknown"
    )

def is_user_replying_to_user(update):
    """Check if the message is a user replying to another user (not the bot)."""
    if not update.message or not update.message.reply_to_message:
        return False
    replied_to_message = update.message.reply_to_message
    bot_id = update.get_bot().id if update.get_bot() else None
    if replied_to_message.from_user and replied_to_message.from_user.id == bot_id:
        return False
    if replied_to_message.from_user and not replied_to_message.from_user.is_bot:
        return True
    return False
