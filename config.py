import os
import json
from dataclasses import dataclass
from typing import Optional

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

class ConfigLoader:
    @staticmethod
    def load_config(bot_name: str, logger=None) -> Optional[BotConfig]:
        try:
            config_path = os.path.join(os.path.dirname(__file__), "bots_config.json")
            with open(config_path, 'r') as f:
                all_configs = json.load(f)
            if bot_name not in all_configs:
                if logger:
                    logger.critical(f"Bot configuration for '{bot_name}' not found.")
                return None
            config_data = all_configs[bot_name]
            config_data['name'] = bot_name
            return BotConfig(**config_data)
        except FileNotFoundError:
            if logger:
                logger.critical("bots_config.json not found.")
            return None
        except json.JSONDecodeError:
            if logger:
                logger.critical("Invalid JSON in bots_config.json.")
            return None
        except TypeError as e:
            if logger:
                logger.critical(f"Invalid configuration structure: {e}")
            return None
