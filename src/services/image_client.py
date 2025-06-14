import aiohttp
import asyncio
from typing import Optional
import logging
import ssl # For ssl.SSLContext
import os # For example usage

from src.config import constants as const # For timeouts, default models etc.

class ImageGenerationClient:
    """
    A client for handling image generation requests to an external API.
    """
    def __init__(self, logger: logging.Logger, ssl_context: ssl.SSLContext,
                 image_provider_name: str = const.IMAGE_PROVIDER_NAME):
        """
        Initializes the ImageGenerationClient.

        Args:
            logger: A logger instance for logging events.
            ssl_context: An SSL context for HTTPS requests.
            image_provider_name: Name of the image provider (for logging/identification).
        """
        self.logger: logging.Logger = logger
        self.ssl_context: ssl.SSLContext = ssl_context
        self.image_provider_name: str = image_provider_name
        # API key is passed directly to the generate_image method for flexibility.

    async def generate_image(self, prompt: str, api_key: str,
                               model_id: Optional[str] = None,
                               image_size: Optional[str] = None,
                               max_prompt_length: Optional[int] = None) -> Optional[str]:
        """
        Generates an image based on the given prompt using an external API.

        Args:
            prompt: The text prompt to generate the image from.
            api_key: The API key for the image generation service.
            model_id: Optional model ID for image generation. Defaults to `const.DEFAULT_IMAGE_MODEL`.
            image_size: Optional image size (e.g., "1024x1024"). Defaults to `const.DEFAULT_IMAGE_SIZE`.
            max_prompt_length: Optional maximum length for the prompt. Defaults to `const.MAX_IMAGE_PROMPT_LENGTH`.

        Returns:
            A string URL to the generated image if successful, otherwise None.
        """
        if not api_key:
            self.logger.error(f"({self.image_provider_name}) API Key for image generation not provided.")
            # Consider raising an error or returning a more specific error message/code
            return None

        # Use provided parameters or fall back to constants/defaults
        current_model_id = model_id or const.DEFAULT_IMAGE_MODEL
        current_image_size = image_size or const.DEFAULT_IMAGE_SIZE
        width, height = current_image_size.split('x')
        current_max_prompt_length = max_prompt_length or const.MAX_IMAGE_PROMPT_LENGTH

        # This example continues to use ModelsLab-like API structure.
        # Adapt api_url and payload if using a different service (e.g., OpenRouter for DALL-E).
        # For OpenRouter DALL-E, the endpoint would be const.OPENROUTER_API_URL
        # and payload would be similar to LLMClient's get_llm_response.
        api_url = "https://modelslab.com/api/v6/images/text2img" # This could be a configurable constant

        headers = {
            "Content-Type": "application/json",
            # Authorization might be needed here if using OpenRouter or similar
            # "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "key": api_key, # Specific to ModelsLab in this example
            "prompt": prompt[:current_max_prompt_length],
            "model_id": current_model_id,
            "samples": "1", # Typically "1" for direct requests
            "height": height,
            "width": width,
            "safety_checker": False, # Configurable
            "seed": None, # Or allow passing a seed
            "base64": False, # False if expecting a URL
            "webhook": None,
            "track_id": None,
        }

        self.logger.info(f"[ImageClient] ({self.image_provider_name}) Requesting image generation for prompt (first 50 chars): '{prompt[:50]}...' with model {current_model_id}")

        try:
            connector = aiohttp.TCPConnector(ssl=self.ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=const.IMAGE_GENERATION_TIMEOUT_SECONDS)
                ) as response:
                    response.raise_for_status() # Raises an error for bad responses (4xx or 5xx)
                    response_json = await response.json()

                    self.logger.info(f"[ImageClient] ({self.image_provider_name}) Full API response: {response_json}")

                    output = response_json.get("output")
                    if output and isinstance(output, list) and len(output) > 0:
                        image_url = output[0]
                        self.logger.info(f"[ImageClient] ({self.image_provider_name}) Successfully generated image. URL: {image_url}")
                        return image_url

                    # Handle cases like "processing" if the API returns such status
                    if response_json.get("status") == "processing" and response_json.get("fetch_result"):
                        self.logger.info(f"[ImageClient] ({self.image_provider_name}) Image is processing. Fetch URL: {response_json['fetch_result']}")
                        # Depending on requirements, you might want to implement polling or return a specific status.
                        # For now, returning None or a placeholder.
                        return None # Or a placeholder like "status:processing"

                    error_detail = response_json.get("error", response_json.get("message", "No image URL or error detail in response."))
                    self.logger.warning(f"[ImageClient] ({self.image_provider_name}) No image URL found or error in response: {error_detail}. Prompt: '{prompt[:50]}...'")
                    return None

        except asyncio.TimeoutError:
            self.logger.error(f"[ImageClient] ({self.image_provider_name}) Image generation request timed out for prompt: '{prompt[:50]}...'")
            return None
        except aiohttp.ClientResponseError as e:
            # Log the response content if available, as it might contain useful error details
            response_text = await e.response.text() if e.response else "No response body."
            self.logger.error(f"[ImageClient] ({self.image_provider_name}) HTTP Error {e.status} for prompt '{prompt[:50]}...': {e.message}. Response: {response_text[:200]}")
            return None
        except Exception as e:
            self.logger.error(f"[ImageClient] ({self.image_provider_name}) General error during image generation for prompt '{prompt[:50]}...': {e}", exc_info=True)
            return None

if __name__ == '__main__':
    # Example Usage (requires running an asyncio event loop)
    async def main():
        # Setup basic logging for the example
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        example_logger = logging.getLogger("ImageClientExample")

        # Dummy SSL context for example
        ssl_ctx = aiohttp.TCPConnector().ssl_context # Not a proper SSL context for production

        image_client = ImageGenerationClient(logger=example_logger, ssl_context=ssl_ctx)

        # IMPORTANT: Replace with a REAL ModelsLab API key or other service key for testing
        test_api_key = os.getenv("MODELSLAB_API_KEY") # Or your actual image API key

        if not test_api_key:
            example_logger.warning("MODELSLAB_API_KEY environment variable not set. Skipping live API call.")
            return

        prompt_text = "A futuristic cityscape at sunset, digital art"
        example_logger.info(f"Attempting to generate image for prompt: '{prompt_text}'")

        image_url_result = await image_client.generate_image(prompt_text, api_key=test_api_key)

        if image_url_result:
            example_logger.info(f"Generated image URL: {image_url_result}")
        else:
            example_logger.error("Failed to generate image.")

    # Python 3.7+
    # asyncio.run(main())
    # For older versions:
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(main())
