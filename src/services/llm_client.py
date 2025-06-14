import aiohttp
import async_timeout
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import List, Dict, Optional

class LLMClient:
    def __init__(self, api_key: str, model: str, bot_username_internal: str, bot_name_display: str, logger, ssl_context):
        self.api_key = api_key
        self.model = model
        self.bot_username_internal = bot_username_internal
        self.bot_name_display = bot_name_display
        self.logger = logger
        self.ssl_context = ssl_context
        self.OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
        self.DEFAULT_MAX_TOKENS_ASK = 1000
        self.DEFAULT_MAX_TOKENS_PERIODIC = 600
        self.TIMEOUT_SECONDS = 30
        self.PERIODIC_TIMEOUT_SECONDS = 60

    async def check_network_health(self) -> bool:
        try:
            async with async_timeout.timeout(5):
                async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=self.ssl_context)) as session:
                    async with session.get("https://openrouter.ai/api/v1/auth/check"):
                        return True
        except Exception as e:
            self.logger.warning(f"Network health check failed: {str(e)}")
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_llm_response(self, chat_id: int, messages: List[Dict], for_periodic_job: bool = False) -> str:
        if not self.api_key:
            self.logger.error(f"[ChatID: {chat_id}] OpenRouter API Key not configured.")
            return "*error beep* My API key is not set up!"

        if not await self.check_network_health():
            self.logger.warning(f"[ChatID: {chat_id}] Network health check failed - skipping LLM call")
            return "*static* My connection is unstable... try again soon."

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": f"https://{self.bot_username_internal}.com",
            "X-Title": f"{self.bot_name_display} Telegram Integration"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": (self.DEFAULT_MAX_TOKENS_PERIODIC if for_periodic_job else self.DEFAULT_MAX_TOKENS_ASK),
            "temperature": 0.7 if for_periodic_job else 0.5,
            "presence_penalty": 0.3,
            "frequency_penalty": 0.3,
            "stream": False
        }
        if for_periodic_job:
            payload["top_p"] = 0.9
        timeout = (self.PERIODIC_TIMEOUT_SECONDS if for_periodic_job else self.TIMEOUT_SECONDS)
        try:
            connector = aiohttp.TCPConnector(ssl=self.ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(
                    self.OPENROUTER_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    response.raise_for_status()
                    response_json = await response.json()
                    self.logger.info(f"[LLM DEBUG] Full LLM API response: {response_json}")
                    choices = response_json.get("choices", [])
                    if choices and choices[0].get("message", {}).get("content"):
                        content = choices[0]["message"]["content"].strip()
                        self.logger.info(f"[LLM DEBUG] Outgoing LLM message: '{content}' (length: {len(content)})")
                        return content
                    else:
                        self.logger.error(f"[ChatID: {chat_id}] Unexpected LLM response format. Full response: {response_json}")
                        return "*error beep* My circuits are fuzzy... try again later!"
        except asyncio.TimeoutError:
            self.logger.error(f"[ChatID: {chat_id}] LLM API request timed out")
            return "*system whine* My processors are overloaded! Try again soon."
        except aiohttp.ClientResponseError as e:
            self.logger.error(f"[ChatID: {chat_id}] LLM API HTTP Error: {e.status}")
            if e.status == 401:
                return "*critical error* My API key is invalid!"
            return "*error beep* An API error occurred. Try again later."
        except Exception as e:
            self.logger.error(f"[ChatID: {chat_id}] LLM Response Error: {e}")
            return "*error beep* An unexpected error occurred."

    async def generate_image_with_openrouter(self, prompt: str) -> Optional[str]:
        api_url = "https://modelslab.com/api/v6/images/text2img"
        headers = {"Content-Type": "application/json"}
        payload = {
            "key": "oyvfspVqJlBV2GNXc5rxnkNZm7Jgyuen4AA9xUTm4NeDMPEGntDfeg9sE7QB",
            "prompt": prompt,
            "model_id": "tamarin-xl-v1",
            "samples": "1",
            "height": "1024",
            "width": "1024",
            "safety_checker": False,
            "seed": None,
            "base64": False,
            "webhook": None,
            "track_id": None,
        }
        try:
            connector = aiohttp.TCPConnector(ssl=self.ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    response.raise_for_status()
                    response_json = await response.json()
                    self.logger.info(f"[DRAW DEBUG] Modelslab response: {response_json}")
                    output = response_json.get("output")
                    if output and isinstance(output, list) and len(output) > 0:
                        return output[0]
                    return None
        except Exception as e:
            self.logger.error(f"Image generation failed: {e}")
            return None
