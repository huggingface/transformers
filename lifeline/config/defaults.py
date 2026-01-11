"""
Default Configuration and Safety Guardrails

Defines safe defaults and validation for Lifeline configuration.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional


logger = logging.getLogger(__name__)


DEFAULT_CONFIG = {
    "version": "0.1.0",

    # Logging configuration
    "log_level": "INFO",
    "log_file": None,  # None = stdout only

    # AI configuration
    "ai": {
        "model_name": "gpt2",
        "use_local": True,
        "alert_threshold": 0.7,
        "suggestion_threshold": 0.5,
        "max_file_size": 1_000_000,  # 1MB
    },

    # Watcher configuration
    "watchers": {
        "file_poll_interval": 2.0,
        "git_poll_interval": 5.0,
        "ignored_patterns": [
            ".git",
            "__pycache__",
            "*.pyc",
            ".lifeline",
            "node_modules",
            ".pytest_cache",
            ".venv",
            "venv",
        ],
    },

    # Safety guardrails
    "safety": {
        "max_memory_size": 100_000_000,  # 100MB
        "max_insights_stored": 2000,
        "max_commits_stored": 1000,
        "auto_save_interval": 300,  # 5 minutes
    },

    # Features
    "features": {
        "proactive_suggestions": True,
        "security_alerts": True,
        "code_quality_analysis": True,
        "commit_analysis": True,
    },
}


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from file or use defaults

    Args:
        config_path: Path to config file (default: .lifeline/config.json)

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path.cwd() / ".lifeline" / "config.json"

    if not config_path.exists():
        logger.info("No config file found, using defaults")
        return DEFAULT_CONFIG.copy()

    try:
        user_config = json.loads(config_path.read_text())
        config = DEFAULT_CONFIG.copy()
        config.update(user_config)

        logger.info(f"Loaded configuration from {config_path}")
        return config

    except Exception as e:
        logger.error(f"Error loading config: {e}")
        logger.info("Using default configuration")
        return DEFAULT_CONFIG.copy()


def save_config(config: Dict[str, Any], config_path: Optional[Path] = None):
    """
    Save configuration to file

    Args:
        config: Configuration dictionary
        config_path: Path to save config (default: .lifeline/config.json)
    """
    if config_path is None:
        config_path = Path.cwd() / ".lifeline" / "config.json"

    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2))

        logger.info(f"Configuration saved to {config_path}")

    except Exception as e:
        logger.error(f"Error saving config: {e}")


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration for safety

    Args:
        config: Configuration to validate

    Returns:
        True if valid, False otherwise
    """
    # Check required keys
    required_keys = ["ai", "watchers", "safety", "features"]

    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required config key: {key}")
            return False

    # Validate safety limits
    safety = config.get("safety", {})

    if safety.get("max_memory_size", 0) > 1_000_000_000:  # 1GB
        logger.error("max_memory_size too large (max: 1GB)")
        return False

    if safety.get("auto_save_interval", 0) < 10:
        logger.error("auto_save_interval too small (min: 10 seconds)")
        return False

    # Validate AI thresholds
    ai = config.get("ai", {})

    if not (0 <= ai.get("alert_threshold", 0.7) <= 1):
        logger.error("alert_threshold must be between 0 and 1")
        return False

    if not (0 <= ai.get("suggestion_threshold", 0.5) <= 1):
        logger.error("suggestion_threshold must be between 0 and 1")
        return False

    return True
