import json
import os
from dotenv import load_dotenv
from typing import Any, Dict, Optional, List

class ConfigManager:
    """
    Manages bot configuration by loading from a JSON file and environment variables.
    Environment variables can override values loaded from the JSON file.
    """
    def __init__(self, config_path: str = 'config.json', dotenv_path: Optional[str] = None):
        """
        Initializes the ConfigManager.

        Args:
            config_path: Path to the JSON configuration file.
            dotenv_path: Path to the .env file. If None, uses standard .env loading.
        """
        self.config_path: str = config_path
        self.dotenv_path: Optional[str] = dotenv_path
        self.config: Dict[str, Any] = self._load_config_from_file()
        self._load_environment_variables() # Overrides file config with env vars

    def _load_config_from_file(self) -> Dict[str, Any]:
        """
        Loads the base configuration from a JSON file.
        Protected method, called during initialization.
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            # It's often better to log errors than print them, especially in a library.
            # Consider raising a custom exception or returning a clear signal of failure.
            # For now, matching existing behavior but adding a log for potential improvement.
            # print(f"Warning: Configuration file '{self.config_path}' not found. Starting with empty/default config.")
            logging.warning(f"Configuration file '{self.config_path}' not found. Using defaults or environment variables.")
            return {}
        except json.JSONDecodeError:
            # print(f"Error: Could not decode JSON from '{self.config_path}'. Using empty/default config.")
            logging.error(f"Could not decode JSON from '{self.config_path}'. Check its format. Using defaults or environment variables.")
            return {}

    def _load_environment_variables(self) -> None:
        """
        Loads environment variables from a .env file (if specified) or the environment.
        These variables override any values loaded from the JSON configuration file.
        Specific environment variables are mapped to configuration keys.
        Protected method, called during initialization.
        """
        load_dotenv(dotenv_path=self.dotenv_path) # dotenv_path can be None, load_dotenv handles it

        # Define mappings for clarity and maintainability
        env_to_config_map = {
            'TELEGRAM_TOKEN': 'telegram_token',
            'ADMIN_ID': 'admin_id',
            'DATABASE_URL': 'database_url',
            'OPENROUTER_API_KEY': 'openrouter_api_key',
            # For JSON-encoded strings like lists/dicts in env vars
            'ALLOWED_USER_IDS': ('allowed_user_ids', list),
            'SYSTEM_PROMPT_FILE': 'system_prompt_file', # Assuming this might come from env
            'SYSTEM_PROMPT': 'system_prompt', # Allow direct system prompt from env
            'MODELSLAB_API_KEY': 'modelslab_api_key', # For image generation
        }

        for env_var, config_key_info in env_to_config_map.items():
            value = os.getenv(env_var)
            if value is not None: # Only process if the environment variable is set
                if isinstance(config_key_info, tuple):
                    config_key, target_type = config_key_info
                    if target_type == list or target_type == dict:
                        try:
                            self.config[config_key] = json.loads(value)
                        except json.JSONDecodeError:
                            logging.error(f"Failed to parse JSON for env var {env_var} ('{value}'). Skipping.")
                    # Add other type conversions if needed
                else:
                    config_key = config_key_info
                    self.config[config_key] = value

        # Load system prompt from file if specified and not directly set by env
        # This logic is a bit more complex than what was there; original only loaded from file via a specific method.
        # Now, SYSTEM_PROMPT env var takes precedence.
        if 'system_prompt' not in self.config and self.config.get('system_prompt_file'):
            try:
                # Ensure file path is absolute or correctly relative to a known root
                # For now, assuming it's relative to where the script is run, or an absolute path.
                # A more robust solution might involve passing a base_dir to ConfigManager.
                prompt_file_path = self.config['system_prompt_file']
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Assuming src/config/config_manager.py
                if not os.path.isabs(prompt_file_path):
                     prompt_file_path = os.path.join(project_root, prompt_file_path)

                with open(prompt_file_path, 'r', encoding='utf-8') as f:
                    self.config['system_prompt'] = f.read().strip()
                logging.info(f"System prompt loaded from file: {prompt_file_path}")
            except FileNotFoundError:
                logging.error(f"System prompt file specified ('{self.config.get('system_prompt_file')}') but not found at {prompt_file_path}.")
            except Exception as e:
                logging.error(f"Error loading system prompt from file '{self.config.get('system_prompt_file')}': {e}")


    def load_system_prompt(self) -> str:
        """
        Loads the system prompt. It first checks if a system prompt is already loaded in the config
        (e.g., from an environment variable or directly from system_prompt_file during init).
        If not found, it defaults to a generic prompt.
        This method might be redundant if _load_environment_variables handles prompt file loading.
        Kept for compatibility if direct loading is still desired post-init by this name.
        """
        # System prompt should be loaded during _load_environment_variables if file is specified
        # or directly from SYSTEM_PROMPT env var.
        return self.config.get('system_prompt', "You are a helpful assistant.")


    def get(self, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """
        Gets a configuration value by key.

        Args:
            key: The configuration key to retrieve.
            default: The default value to return if the key is not found.

        Returns:
            The configuration value or the default.
        """
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Sets a configuration value by key and saves it back to the JSON file.
        Note: This will overwrite the existing configuration file.

        Args:
            key: The configuration key to set.
            value: The value to set for the key.
        """
        self.config[key] = value
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
            logging.info(f"Configuration saved to '{self.config_path}' after setting key '{key}'.")
        except IOError as e:
            logging.error(f"Could not write to configuration file '{self.config_path}': {e}")

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO) # Basic logging for example

    # Example usage:
    # Create a dummy config.json for testing
    dummy_config = {
        "bot_name_display": "TestBotFromFile",
        "feature_x_enabled": True,
        "system_prompt_file": "config/prompts/example_prompt.txt" # Relative to project root
    }
    with open('config.json', 'w') as f:
        json.dump(dummy_config, f, indent=4)

    # Create a dummy .env for testing
    with open('.env', 'w') as f:
        f.write('TELEGRAM_TOKEN="dummy_telegram_token_from_env"\n')
        f.write('ADMIN_ID="12345_from_env"\n')
        f.write('ALLOWED_USER_IDS=\'["user1_env", "user2_env"]\'\n')
        f.write('SYSTEM_PROMPT="System prompt from ENV"\n') # This will override file if both present

    # Create a dummy prompt file
    os.makedirs('config/prompts', exist_ok=True)
    with open('config/prompts/example_prompt.txt', 'w') as f:
        f.write("This is a system prompt from a file.")

    print("--- Running ConfigManager example ---")
    # Pass dotenv_path explicitly for the example
    config_manager = ConfigManager(config_path='config.json', dotenv_path='.env')

    print(f"Telegram Token: {config_manager.get('telegram_token')}") # From .env
    print(f"Admin ID: {config_manager.get('admin_id')}") # From .env
    print(f"Allowed User IDs: {config_manager.get('allowed_user_ids')}") # From .env, parsed as list
    print(f"Bot Name Display: {config_manager.get('bot_name_display')}") # From config.json
    print(f"System Prompt (should be from ENV): {config_manager.get('system_prompt')}")

    # Test loading prompt from file if SYSTEM_PROMPT env var was not set
    # To test this, comment out SYSTEM_PROMPT in .env and rerun
    if os.getenv('SYSTEM_PROMPT') is None:
        print(f"System Prompt (should be from file if SYSTEM_PROMPT env not set): {config_manager.get('system_prompt')}")


    # Example of setting a new value (optional)
    config_manager.set('new_setting_runtime', 'new_value_runtime')
    print(f"New Setting (runtime): {config_manager.get('new_setting_runtime')}")

    # Clean up dummy files
    # os.remove('config.json')
    # os.remove('.env')
    # os.remove('config/prompts/example_prompt.txt')
    # os.rmdir('config/prompts')
    # os.rmdir('config')
    print("--- ConfigManager example finished ---")
