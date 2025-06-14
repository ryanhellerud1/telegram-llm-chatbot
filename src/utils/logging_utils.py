import logging
import os
import sys

def setup_bot_logging(
    logger_name: str,
    log_level_env_key: str,
    bot_display_name_or_env_key: str,
    default_log_level: str = "INFO"
) -> logging.Logger:
    """
    Configures and returns a logger instance, typically for a bot.

    This function sets up a logger with a specified name, log level (derived from
    an environment variable or a default), and a custom log format that includes
    a display name for the bot (also potentially from an environment variable).
    It also configures a console handler for the logger and, if no handlers are
    configured on the root logger, it adds a similarly formatted console handler
    to the root logger to catch logs from other libraries.

    Args:
        logger_name: The internal name for the logger (e.g., "MyBot").
        log_level_env_key: The environment variable key that stores the desired
                           log level string (e.g., "MY_BOT_LOG_LEVEL").
        bot_display_name_or_env_key: The string to be used as the bot's display name in logs.
                                     If an environment variable exists with this key, its value
                                     will be used as the display name. Otherwise, this string
                                     itself is used as the display name.
        default_log_level: The default log level (e.g., "INFO") if the environment
                           variable specified by `log_level_env_key` is not set or invalid.

    Returns:
        A configured `logging.Logger` instance.
    """
    logger = logging.getLogger(logger_name)

    # Determine log level from environment variable or use default
    log_level_str = os.getenv(log_level_env_key, default_log_level).upper()
    log_level = getattr(logging, log_level_str, logging.INFO) # Fallback to INFO if getattr fails

    logger.setLevel(log_level) # Set level for the specific bot logger

    # Configure root logger's level and add a handler if it has none.
    # This helps in capturing and formatting logs from libraries used by the bot.
    root_logger = logging.getLogger()
    if not root_logger.hasHandlers():
        root_logger.setLevel(log_level) # Set root logger to the same level

    # Determine bot display name: use env var if `bot_display_name_or_env_key` is a key AND it's set,
    # otherwise use `bot_display_name_or_env_key` as the literal display name.
    bot_display_name = os.getenv(bot_display_name_or_env_key, bot_display_name_or_env_key)
    # If os.getenv returns the key itself (meaning env var not found), then key is the name.
    # If bot_display_name is empty after this, fallback to logger_name.
    if not bot_display_name:
        bot_display_name = logger_name

    formatter = logging.Formatter(
        f'%(asctime)s - [{bot_display_name}] - %(levelname)s - [%(name)s] - %(message)s'
    )

    # Add console handler to the specific bot logger if it doesn't have one
    if not any(isinstance(h, logging.StreamHandler) and h.stream == sys.stdout for h in logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add console handler to the root logger if it doesn't have one, using the same format.
    if not root_logger.hasHandlers():
        root_console_handler = logging.StreamHandler(sys.stdout)
        root_console_handler.setFormatter(formatter)
        root_logger.addHandler(root_console_handler)

    # Suppress overly verbose library logging.
    # Consider making these configurable if more flexibility is needed.
    libraries_to_adjust = {
        "httpx": logging.WARNING,
        "telegram": logging.INFO, # telegram.ext can be verbose at DEBUG
        "apscheduler": logging.WARNING,
        "httpcore": logging.WARNING, # Often very verbose at INFO/DEBUG
    }
    for lib_name, lib_level in libraries_to_adjust.items():
        logging.getLogger(lib_name).setLevel(lib_level)

    logger.info(f"Logging configured for '{bot_display_name}' (logger name: '{logger_name}'). Effective level: {log_level_str}")

    return logger

if __name__ == '__main__':
    # Example Usage:
    # To run this example:
    # 1. Optionally set environment variables:
    #    export BOT_A_LOG_LEVEL=DEBUG
    #    export BOT_A_DISPLAY_NAME="AwesomeBotA"
    #    export BOT_B_LOG_LEVEL=INFO

    print("\n--- Example 1: Bot A (using env vars for display name and level) ---")
    bot_a_logger = setup_bot_logging(
        logger_name="BotAInternal",
        log_level_env_key="BOT_A_LOG_LEVEL",
        bot_display_name_or_env_key="BOT_A_DISPLAY_NAME", # Tries to get BOT_A_DISPLAY_NAME from env
        default_log_level="INFO"
    )
    bot_a_logger.debug("This is a debug message from Bot A.")
    bot_a_logger.info("This is an info message from Bot A.")
    bot_a_logger.warning("This is a warning from Bot A.")

    print("\n--- Example 2: Bot B (using literal display name, default INFO level if BOT_B_LOG_LEVEL not set) ---")
    bot_b_logger = setup_bot_logging(
        logger_name="BotBInternal",
        log_level_env_key="BOT_B_LOG_LEVEL", # Will default to INFO if not set
        bot_display_name_or_env_key="CoolBotB" # Literal name, as BOT_B_DISPLAY_NAME is not set
    )
    bot_b_logger.debug("This debug message from Bot B should NOT appear (unless BOT_B_LOG_LEVEL=DEBUG).")
    bot_b_logger.info("This is an info message from Bot B.")

    print("\n--- Example 3: Testing library log suppression ---")
    logging.getLogger("httpx").info("This httpx info message should NOT appear.")
    logging.getLogger("httpx").warning("This httpx warning message SHOULD appear.")
    logging.getLogger("telegram.ext.Application").debug("A very verbose telegram debug log - should not appear.")
    logging.getLogger("telegram.ext.Application").info("An info telegram log - should appear.")


    print("\n--- Example 4: Root logger test ---")
    # Logs from other modules that use the root logger without specific configuration
    # will now also use the formatter set up by setup_bot_logging (if it was the first to configure root).
    root_test_logger = logging.getLogger("some.other.module")
    root_test_logger.info("This message from 'some.other.module' should be formatted by our root handler.")

    print("\nCheck console output for formatted log messages and verbosity based on settings.")
