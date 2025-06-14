# Kovu Telegram Bot (LLM-Driven)

A Telegram bot named **Kovu**: a cybernetically enhanced, AI-powered red Siberian husky designed to provide strategic advice, encouragement, and community support for the $Kovu memecoin community. The bot leverages OpenRouter's LLM API for intelligent, context-aware responses and periodic group engagement.

## Features
- **/startkovu**: Starts periodic LLM-driven responses in the chat (every 30 seconds by default).
- **/ask [question]**: Directly ask Kovu a question and receive an LLM-generated answer.
- **Welcomes new members**: Greets new users joining the chat.
- **Message history**: Maintains a rolling history of recent messages for context-aware replies.
- **Keyword handler**: (Optional) Responds to certain keywords (currently commented out).
- **Strict persona**: Kovu always responds as a helpful, optimistic, first-person husky with no markdown formatting.

## Setup
1. **Clone the repository** and navigate to the project directory.
2. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```
3. **Create a `.env` file** in the project directory with the following variables:
   ```env
   KOVU_BOT_TOKEN=your_telegram_bot_token
   OPENROUTER_API_KEY=your_openrouter_api_key
   ```
4. **Run the bot**:
   ```sh
   python kovu_bot_llm.py
   ```

## Environment Variables
- `KOVU_BOT_TOKEN`: Telegram bot token from BotFather.
- `OPENROUTER_API_KEY`: API key for OpenRouter (https://openrouter.ai/).

## File Overview
- `kovu_bot_llm.py`: Main bot logic, handlers, and LLM integration.
- `requirements.txt`: Python dependencies.
- `.env`: (Not committed) Your secrets for bot and API access.

## Notes
- The bot uses a periodic job to check for new messages and respond if there is new activity.
- All LLM prompts and persona constraints are hardcoded for Kovu's unique style.
- Logging is enabled for debugging and monitoring.

## License
MIT License (or specify your own)

---
*For questions or support, contact the project maintainer.*
