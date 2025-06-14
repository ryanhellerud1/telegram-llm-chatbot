import aiohttp
import async_timeout
import asyncio
import ssl # For ssl.SSLContext
import logging # For logging.Logger
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from typing import List, Dict, Optional, Any # Added Any
from src.config import constants as const
from src.services.chat_data import ChatHistoryManager, ChatData # Added ChatData
from src.config.config_manager import ConfigManager

class LLMClient:
    """
    Client for interacting with a Large Language Model (LLM) API,
    including preparing messages, handling API calls, and retries.
    """
    def __init__(self,
                 api_key: str,
                 model: str,
                 bot_username_internal: str,
                 bot_name_display: str,
                 logger: logging.Logger,
                 ssl_context: ssl.SSLContext,
                 config_manager: ConfigManager,
                 openrouter_api_key: Optional[str] = None):
        """
        Initializes the LLMClient.

        Args:
            api_key: The primary API key for the LLM service (might be general or specific if not OpenRouter).
            model: The LLM model identifier to be used.
            bot_username_internal: Internal username of the bot (e.g., for API headers).
            bot_name_display: Display name of the bot (e.g., for API headers or logging).
            logger: A logger instance for logging events.
            ssl_context: An SSL context for HTTPS requests.
            config_manager: An instance of ConfigManager to access configuration like system prompts.
            openrouter_api_key: Optional specific API key for OpenRouter. If None, `api_key` is used.
        """
        self.api_key: str = api_key
        self.model: str = model
        self.bot_username_internal: str = bot_username_internal
        self.bot_name_display: str = bot_name_display
        self.logger: logging.Logger = logger
        self.ssl_context: ssl.SSLContext = ssl_context
        self.config_manager: ConfigManager = config_manager
        self.openrouter_api_key: str = openrouter_api_key or api_key # Use primary if specific not given

    def prepare_messages_for_llm(self,
                                 chat_id: int,
                                 chat_history_manager: ChatHistoryManager,
                                 include_introduction: bool = False) -> List[Dict[str, str]]:
        """
        Prepares a list of messages from chat history for LLM consumption.
        Optionally prepends a system introduction/prompt.

        Args:
            chat_id: The ID of the chat whose history is to be prepared.
            chat_history_manager: The manager instance to retrieve chat history.
            include_introduction: Whether to prepend the system prompt.

        Returns:
            A list of message dictionaries, each with "role" and "content".
            Returns an empty list if no relevant history or prompt is found.
        """
        messages: List[Dict[str, str]] = []
        if include_introduction:
            system_prompt = self.config_manager.get('system_prompt', "You are a helpful assistant.")
            messages.append({"role": "system", "content": system_prompt})

        chat_data: Optional[ChatData] = chat_history_manager.get_chat_data(chat_id)
        if chat_data and chat_data.message_history: # message_history is already a list of dicts
            for msg in chat_data.message_history:
                # Ensure only 'role' and 'content' are passed, and they are strings.
                # This assumes msg structure is {'role': str, 'content': str, ...}
                if isinstance(msg.get("role"), str) and isinstance(msg.get("content"), str):
                    messages.append({"role": msg["role"], "content": msg["content"]})
                else:
                    self.logger.warning(f"[ChatID: {chat_id}] Skipping malformed message in history: {msg}")

        # If only system prompt is present and no user/assistant messages,
        # it might not be useful to send to LLM depending on the model.
        # However, current logic allows it.
        if not messages and not include_introduction:
             self.logger.info(f"[ChatID: {chat_id}] No messages to prepare (history empty and no introduction).")
        elif len(messages) == (1 if include_introduction else 0):
             self.logger.info(f"[ChatID: {chat_id}] Prepared messages consist only of system prompt or are empty.")

        return messages

    async def check_network_health(self) -> bool:
        """
        Performs a basic check to see if the OpenRouter API authentication endpoint is reachable.
        This can indicate general network connectivity to the service.

        Returns:
            True if the auth check endpoint returns a successful (e.g., 200) response, False otherwise.
        """
        # Construct auth check URL carefully, ensure it's correct for the provider
        # For OpenRouter, if base URL is "https://openrouter.ai/api/v1", then "/auth/check" is appended.
        auth_check_url = const.OPENROUTER_API_URL.replace("/chat/completions", "/auth/check")

        try:
            # Use a short timeout for health checks
            timeout = aiohttp.ClientTimeout(total=5.0) # 5 seconds
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=self.ssl_context)) as session:
                async with session.get(auth_check_url, timeout=timeout) as response:
                    # Consider any 2xx or 3xx status as healthy for a simple check
                    # OpenRouter's /auth/check returns 200 if key is valid, 401 if invalid, but endpoint is up.
                    # For a pure network health check, any response means it's up.
                    # For now, let's assume any response means network is okay.
                    # response.raise_for_status() # This would fail on 401 from /auth/check
                    self.logger.info(f"Network health check to {auth_check_url} status: {response.status}")
                    return True # Endpoint is reachable
        except aiohttp.ClientConnectorError as e:
            self.logger.warning(f"Network health check failed: Connection error to {auth_check_url} - {e}")
            return False
        except asyncio.TimeoutError:
            self.logger.warning(f"Network health check failed: Request to {auth_check_url} timed out.")
            return False
        except Exception as e:
            self.logger.error(f"Network health check to {auth_check_url} failed with unexpected error: {e}", exc_info=True)
            return False

    @retry(stop=stop_after_attempt(const.RETRY_ATTEMPTS),
           wait=wait_exponential(multiplier=const.RETRY_WAIT_EXPONENTIAL_MULTIPLIER,
                                 max=const.RETRY_WAIT_EXPONENTIAL_MAX))
    async def get_llm_response(self,
                               chat_id: int,
                               messages: List[Dict[str, str]],
                               for_periodic_job: bool = False) -> str:
        """
        Gets a response from the LLM API for the given messages.
        Includes retry logic for transient errors.

        Args:
            chat_id: Identifier for the chat, used for logging.
            messages: A list of message dictionaries to send to the LLM.
            for_periodic_job: Flag indicating if this request is for a periodic job,
                              which might use different LLM parameters (e.g., max_tokens, temperature).

        Returns:
            The LLM's response content as a string, or an error message string if an issue occurs.
        """
        if not self.openrouter_api_key: # Check specific OpenRouter key
            self.logger.error(f"[ChatID: {chat_id}] OpenRouter API Key not configured for LLM.")
            return "*error beep* My OpenRouter API key is not set up!"

        # Network health check might need to be adapted if it's OpenRouter specific
        if not await self.check_network_health():
            self.logger.warning(f"[ChatID: {chat_id}] Network health check failed - skipping LLM call")
            return "*static* My connection is unstable... try again soon."

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openrouter_api_key}", # Use OpenRouter key
            "HTTP-Referer": const.OPENROUTER_REFERRER, # Use constant
            "X-Title": f"{self.bot_name_display} Telegram Integration" # This seems fine
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": (const.DEFAULT_MAX_TOKENS_PERIODIC if for_periodic_job else const.DEFAULT_MAX_TOKENS_ASK),
            "temperature": 0.7 if for_periodic_job else 0.5, # These could be constants too
            "presence_penalty": 0.3, # These could be constants too
            "frequency_penalty": 0.3, # These could be constants too
            "stream": False
        }
        if for_periodic_job:
            payload["top_p"] = 0.9 # This could be a constant
        timeout_seconds = (const.PERIODIC_TIMEOUT_SECONDS if for_periodic_job else const.TIMEOUT_SECONDS)
        try:
            connector = aiohttp.TCPConnector(ssl=self.ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(
                    const.OPENROUTER_API_URL, # Use constant
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout_seconds)
                ) as response:
                    response.raise_for_status()
                    response_json = await response.json()
                    self.logger.info(f"[LLM DEBUG] ({const.PROVIDER_NAME}) Full API response: {response_json}")
                    choices = response_json.get("choices", [])
                    if choices and choices[0].get("message", {}).get("content"):
                        content = choices[0]["message"]["content"].strip()
                        self.logger.info(f"[LLM DEBUG] ({const.PROVIDER_NAME}) Outgoing message: '{content}' (length: {len(content)})")
                        return content
                    else:
                        self.logger.error(f"[ChatID: {chat_id}] ({const.PROVIDER_NAME}) Unexpected response format. Full response: {response_json}")
                        return "*error beep* My circuits are fuzzy... try again later!"
        except asyncio.TimeoutError:
            self.logger.error(f"[ChatID: {chat_id}] ({const.PROVIDER_NAME}) API request timed out")
            return "*system whine* My processors are overloaded! Try again soon."
        except aiohttp.ClientResponseError as e:
            self.logger.error(f"[ChatID: {chat_id}] ({const.PROVIDER_NAME}) API HTTP Error: {e.status}")
            if e.status == 401: # Unauthorized
                return f"*critical error* My {const.PROVIDER_NAME} API key is invalid!"
            return "*error beep* An API error occurred. Try again later."
        except Exception as e:
            self.logger.error(f"[ChatID: {chat_id}] ({const.PROVIDER_NAME}) Response Error: {e}")
            return "*error beep* An unexpected error occurred."
