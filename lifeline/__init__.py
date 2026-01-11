"""
Lifeline - A Living AI Daemon for Transformers

Lifeline transforms the Transformers library into a continuously aware,
proactive AI companion that monitors, learns, and assists autonomously.

✨ Always watching, always learning, always ready to help ✨
"""

__version__ = "0.1.0"
__author__ = "The Lifeline Collective"

from lifeline.core.daemon import LifelineDaemon
from lifeline.cli.interface import LifelineCLI

__all__ = ["LifelineDaemon", "LifelineCLI"]
