# General Telegram LLM Bot Framework

A flexible Telegram bot framework supporting multiple personalities (e.g., Kovu, Kenji, Narly) with LLM-driven responses, periodic engagement, and image generation. The bot leverages OpenRouter's LLM API for intelligent, context-aware replies and can be configured for different personas via a config file.

## Features
- **/start**: Starts periodic LLM-driven responses in the chat (every 30 seconds by default).
- **/ask [question]**: Directly ask the bot a question and receive an LLM-generated answer.
- **/draw [prompt]**: Generate an image from a text prompt (uses Modelslab API).
- **Welcomes new members**: Greets new users joining the chat.
- **Message history**: Maintains a rolling history of recent messages for context-aware replies.
- **Keyword handler**: (Optional) Responds to certain keywords.
- **Multi-bot support**: Easily run different bots/personalities from a single codebase.

## Setup
1. **Clone the repository** and navigate to the project directory.
2. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```
3. **Configure your bots**:
   - Edit `bots_config.json` to define each bot's settings (see example in the file).
   - Place your system prompt files in the `prompts/` directory.
4. **Create a `.env` file** in the project directory with the following variables for each bot (prefix matches `env_prefix` in `bots_config.json`):
   ```env
   KOVU_BOT_TOKEN=your_telegram_bot_token_for_kovu
   KOVU_OPENROUTER_API_KEY=your_openrouter_api_key
   KENJI_BOT_TOKEN=your_telegram_bot_token_for_kenji
   KENJI_OPENROUTER_API_KEY=your_openrouter_api_key
   NARLY_BOT_TOKEN=your_telegram_bot_token_for_narly
   NARLY_OPENROUTER_API_KEY=your_openrouter_api_key
   # ...add more as needed
   ```
   - You may also set a global `OPENROUTER_API_KEY` as a fallback.

5. **Run the bot** (specify which bot to run):
   ```sh
   python general_bot.py --bot <botname>
   ```
   - `<botname>` must match a key in `bots_config.json` (e.g., `kovubot`, `kenjibot`, `narlybot`).
   - Example:
     ```sh
     python general_bot.py --bot kenjibot
     ```

## Environment Variables
- `<PREFIX>BOT_TOKEN`: Telegram bot token for each bot (e.g., `KOVU_BOT_TOKEN`).
- `<PREFIX>OPENROUTER_API_KEY`: API key for OpenRouter for each bot (e.g., `KENJI_OPENROUTER_API_KEY`).
- `OPENROUTER_API_KEY`: (Optional) Global fallback API key for OpenRouter.

## File Overview
- `general_bot.py`: Main multi-bot logic, handlers, and LLM/image integration.
- `bots_config.json`: Configuration for all supported bots/personalities.
- `prompts/`: System prompt files for each bot.
- `requirements.txt`: Python dependencies.
- `.env`: (Not committed) Your secrets for bot and API access.

## Notes
- The bot uses a periodic job to check for new messages and respond if there is new activity.
- All LLM prompts and persona constraints are loaded from the config and prompt files.
- Logging is enabled for debugging and monitoring.
- You can run multiple bots (in separate processes) by specifying different `--bot` arguments.

## License
MIT License (or specify your own)

---
*For questions or support, contact the project maintainer.*
