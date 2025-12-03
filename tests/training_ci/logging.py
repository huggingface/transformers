"""Logging utilities for training CI tests."""

import logging
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


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors based on log level."""

    LEVEL_COLORS = {
        logging.DEBUG: Colors.DIM + Colors.CYAN,
        logging.INFO: Colors.WHITE,
        logging.WARNING: Colors.BRIGHT_YELLOW,
        logging.ERROR: Colors.BRIGHT_RED,
        logging.CRITICAL: Colors.BOLD + Colors.BRIGHT_RED,
    }

    def __init__(self, fmt: str = None, datefmt: str = None):
        super().__init__(fmt, datefmt)

    # Loggers that should be dimmed (less important/verbose)
    DIMMED_LOGGERS = {"httpx", "httpcore", "urllib3", "requests"}

    def format(self, record: logging.LogRecord) -> str:
        # Check if this logger should be dimmed
        is_dimmed = record.name in self.DIMMED_LOGGERS

        if is_dimmed:
            # Dim the entire log line for httpx and similar
            timestamp = self.formatTime(record, self.datefmt)
            message = record.getMessage()
            return (
                f"{Colors.DIM}{timestamp} - {record.name} - {record.levelname:8} - "
                f"{message}{Colors.RESET}"
            )

        # Get color for this level
        color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)

        # Color the level name
        levelname = record.levelname
        colored_levelname = f"{color}{levelname:8}{Colors.RESET}"

        # Color the timestamp
        colored_time = f"{Colors.DIM}{self.formatTime(record, self.datefmt)}{Colors.RESET}"

        # Color the logger name
        colored_name = f"{Colors.BLUE}{record.name}{Colors.RESET}"

        # Get message (no number highlighting)
        message = record.getMessage()

        return f"{colored_time} - {colored_name} - {colored_levelname} - {message}"


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
