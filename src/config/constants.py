# Constants for TelegramBot
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MAX_TOKENS_ASK = 1000
DEFAULT_MAX_TOKENS_PERIODIC = 600
TIMEOUT_SECONDS = 30
PERIODIC_TIMEOUT_SECONDS = 60
HISTORY_MAX_LENGTH = 20
PERIODIC_JOB_INTERVAL_SECONDS = 30
PERIODIC_JOB_FIRST_RUN_DELAY_SECONDS = 5

# Constants for LLMClient
DEFAULT_MAX_TOKENS = 1500  # Default max tokens for LLMClient
RETRY_ATTEMPTS = 3
RETRY_WAIT_EXPONENTIAL_MULTIPLIER = 1
RETRY_WAIT_EXPONENTIAL_MAX = 60
REQUEST_TIMEOUT = 30  # Timeout for API requests in seconds
IMAGE_GENERATION_TIMEOUT_SECONDS = 120 # Timeout for image generation in seconds
OPENROUTER_REFERRER = "https://github.com/TrueOpenVR/TrueOpenVR-driver" # Referrer for OpenRouter API
PROVIDER_NAME = "OpenRouter" # Provider name for LLMClient
IMAGE_PROVIDER_NAME = "OpenRouter" # Provider name for image generation
DEFAULT_IMAGE_MODEL = "openai/dall-e-3" # Default image model
DEFAULT_IMAGE_SIZE = "1024x1024" # Default image size
MAX_IMAGE_PROMPT_LENGTH = 4000 # Max length for image prompt (bytes for dall-e-3)
