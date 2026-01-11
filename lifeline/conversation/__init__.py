"""
Conversation Module - Natural Language Interface for Lifeline

Enables natural language conversation with transformers!
"""

from lifeline.conversation.voice import TransformerVoice, ConversationManager
from lifeline.conversation.chat_interface import InteractiveChatSession, start_chat_session

__all__ = [
    "TransformerVoice",
    "ConversationManager",
    "InteractiveChatSession",
    "start_chat_session",
]
