"""Logging utilities for training CI tests."""

import logging
import re
import sys

logger = logging.getLogger()

_logged: set[str] = set()


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright variants
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_CYAN = "\033[96m"


# Regex to match numbers (integers, floats, percentages, with optional commas)
NUMBER_PATTERN = re.compile(r'(\d[\d,]*\.?\d*%?|\d+\.?\d*[a-zA-Z]*)')


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors based on log level and highlights numbers."""

    LEVEL_COLORS = {
        logging.DEBUG: Colors.DIM + Colors.CYAN,
        logging.INFO: Colors.WHITE,  # Default white for INFO, numbers get green
        logging.WARNING: Colors.BRIGHT_YELLOW,
        logging.ERROR: Colors.BRIGHT_RED,
        logging.CRITICAL: Colors.BOLD + Colors.BRIGHT_RED,
    }

    def __init__(self, fmt: str = None, datefmt: str = None):
        super().__init__(fmt, datefmt)

    def _highlight_numbers(self, text: str) -> str:
        """Highlight numbers in green within the text."""
        return NUMBER_PATTERN.sub(
            f"{Colors.BRIGHT_GREEN}\\1{Colors.RESET}",
            text
        )

    def format(self, record: logging.LogRecord) -> str:
        # Get color for this level
        color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)

        # Color the level name
        levelname = record.levelname
        colored_levelname = f"{color}{levelname:8}{Colors.RESET}"

        # Color the timestamp
        colored_time = f"{Colors.DIM}{self.formatTime(record, self.datefmt)}{Colors.RESET}"

        # Color the logger name
        colored_name = f"{Colors.BLUE}{record.name}{Colors.RESET}"

        # Get message and highlight numbers in green
        message = record.getMessage()
        if record.levelno == logging.INFO:
            # For INFO: white text with green numbers
            colored_message = self._highlight_numbers(message)
        else:
            # For other levels: use level color for entire message
            colored_message = f"{color}{message}{Colors.RESET}"

        return f"{colored_time} - {colored_name} - {colored_levelname} - {colored_message}"


def init_logger() -> None:
    """Initialize the global logger with colored stdout handler and INFO level."""
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    # Use colored formatter if terminal supports it, plain otherwise
    if sys.stdout.isatty():
        formatter = ColoredFormatter(datefmt="%Y-%m-%d %H:%M:%S")
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    ch.setFormatter(formatter)
    logger.addHandler(ch)


def warn_once(logger: logging.Logger, msg: str) -> None:
    """Log a warning message only once per unique message.

    Uses a global set to track messages that have already been logged
    to prevent duplicate warning messages from cluttering the output.

    Args:
        logger: The logger instance to use for warning.
        msg: The warning message to log.
    """
    if msg not in _logged:
        logger.warning(msg)
        _logged.add(msg)
