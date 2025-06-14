from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime

@dataclass
class ChatData:
    """Data structure for chat room information."""
    message_history: List[Dict[str, Any]] = field(default_factory=list)
    last_processed_history_len: int = 0
    new_activity_flag: bool = False
    created_at: datetime = field(default_factory=datetime.now)

class ChatHistoryManager:
    def __init__(self, history_max_length: int = 20):
        self.chat_rooms = {}
        self.HISTORY_MAX_LENGTH = history_max_length

    def ensure_chat_data(self, chat_id: int) -> ChatData:
        if chat_id not in self.chat_rooms:
            self.chat_rooms[chat_id] = ChatData()
        return self.chat_rooms[chat_id]

    def add_message_to_history(self, chat_id: int, role: str, content: str, user_id: int, username: str = "Unknown", is_reply_to_user: bool = False, logger=None):
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
        if not is_reply_to_user:
            chat_data.new_activity_flag = True
        if logger:
            logger.info(f"[ChatID: {chat_id}] Message added (role: {role}, reply_to_user: {is_reply_to_user}). History length: {len(chat_data.message_history)}")

    def get_last_active_user(self, chat_id: int) -> str:
        chat_data = self.chat_rooms.get(chat_id)
        if not chat_data or not chat_data.message_history:
            return None
        for msg in reversed(chat_data.message_history):
            if msg.get('role') == 'user' and msg.get('username'):
                return msg.get('username')
        return None

    def get_chat_data(self, chat_id: int) -> ChatData:
        """Return the ChatData for a given chat_id, or None if not found."""
        return self.chat_rooms.get(chat_id)
