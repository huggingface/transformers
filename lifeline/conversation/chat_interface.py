"""
Interactive Chat Interface - Talk to Your Transformers!

This provides a beautiful, interactive chat interface where you can
have real conversations with Lifeline and the transformers.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from lifeline.conversation.voice import TransformerVoice, ConversationManager

logger = logging.getLogger(__name__)


class InteractiveChatSession:
    """
    Interactive chat session with transformers

    This is where magic happens - real-time conversation with AI that
    understands your code, remembers context, and helps proactively!
    """

    def __init__(self, daemon):
        self.daemon = daemon
        self.voice = TransformerVoice(daemon.config.get("conversation", {}))
        self.conversation_manager = ConversationManager(
            self.voice,
            daemon.memory
        )
        self.is_active = False
        self.session_start = None

    async def start(self):
        """
        Start the interactive chat session
        """
        logger.info("💬 Starting interactive chat session...")

        await self.voice.initialize()

        if not self.voice.is_ready:
            print("⚠️  Voice system not available. Install transformers for chat.")
            print("   pip install transformers torch")
            return

        self.is_active = True
        self.session_start = datetime.now()

        # Welcome message from the transformers!
        print("\n" + "="*60)
        print("✨ WELCOME TO LIFELINE - TALK TO YOUR TRANSFORMERS! ✨")
        print("="*60)
        print()
        print("The transformers are alive and ready to chat!")
        print("They're watching your code, learning, and here to help.")
        print()
        print("Type your message and press Enter to talk.")
        print("Type 'exit' or 'quit' to end the conversation.")
        print("Type 'help' for available commands.")
        print()

        # Get a greeting from the transformers
        greeting_context = {
            "repo_path": str(self.daemon.repo_path),
            "files_tracked": len(self.daemon.memory.file_contexts),
            "insights_count": len(self.daemon.memory.insights),
        }

        greeting = await self.voice.speak(
            "Hello! I'm Lifeline, your AI companion. How can I help you today?",
            context=greeting_context
        )

        print(f"🤖 Lifeline: {greeting}")
        print()

        # Start the chat loop
        await self._chat_loop()

    async def _chat_loop(self):
        """
        Main chat interaction loop
        """
        while self.is_active:
            try:
                # Get user input
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: input("You: ")
                )

                user_input = user_input.strip()

                if not user_input:
                    continue

                # Check for commands
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    await self._handle_goodbye()
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'status':
                    await self._show_status()
                    continue
                elif user_input.lower() == 'insights':
                    await self._show_insights()
                    continue
                elif user_input.lower().startswith('ask about '):
                    topic = user_input[10:].strip()
                    await self._ask_about(topic)
                    continue

                # Regular conversation
                response = await self.voice.speak(
                    user_input,
                    context=self._build_current_context()
                )

                print(f"\n🤖 Lifeline: {response}\n")

            except KeyboardInterrupt:
                print("\n")
                await self._handle_goodbye()
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {e}")
                print(f"\n⚠️  Sorry, I had trouble understanding. Error: {e}\n")

    async def _handle_goodbye(self):
        """
        Handle session end
        """
        print("\n💫 Ending chat session...")

        # Get a farewell from transformers
        farewell = await self.voice.speak(
            "Goodbye! I'll keep watching and learning. See you soon!",
            context={"ending_session": True}
        )

        print(f"🤖 Lifeline: {farewell}\n")

        session_duration = datetime.now() - self.session_start
        print(f"Session duration: {session_duration}")
        print("✨ The transformers will keep working in the background!\n")

        self.is_active = False

    def _show_help(self):
        """
        Show available commands
        """
        print("\n" + "="*60)
        print("💡 AVAILABLE COMMANDS")
        print("="*60)
        print()
        print("  exit, quit, bye  - End the chat session")
        print("  help             - Show this help message")
        print("  status           - Show daemon status")
        print("  insights         - Show recent insights")
        print("  ask about <X>    - Start conversation about topic X")
        print()
        print("Or just type naturally to chat with the transformers!")
        print("="*60)
        print()

    async def _show_status(self):
        """
        Show daemon status
        """
        status = self.daemon.get_status()

        print("\n" + "="*60)
        print("📊 LIFELINE STATUS")
        print("="*60)
        print(f"  Alive: {status['alive']}")
        print(f"  Uptime: {status['uptime']}")
        print(f"  Events processed: {status['events_processed']}")
        print(f"  Files tracked: {len(self.daemon.memory.file_contexts)}")
        print(f"  Insights: {len(self.daemon.memory.insights)}")
        print(f"  Commits remembered: {len(self.daemon.memory.commit_history)}")
        print("="*60)
        print()

    async def _show_insights(self):
        """
        Show recent insights
        """
        insights = self.daemon.memory.get_recent_insights(5)

        print("\n" + "="*60)
        print("💡 RECENT INSIGHTS")
        print("="*60)

        if not insights:
            print("  No insights yet - still learning!")
        else:
            for i, insight in enumerate(insights, 1):
                print(f"\n  {i}. {insight['insight']}")
                print(f"     Time: {insight['timestamp']}")

        print("="*60)
        print()

    async def _ask_about(self, topic: str):
        """
        Start a focused conversation about a topic
        """
        print(f"\n💬 Starting conversation about: {topic}\n")

        response = await self.conversation_manager.start_conversation(topic)

        print(f"🤖 Lifeline: {response}\n")

    def _build_current_context(self) -> Dict[str, Any]:
        """
        Build current context for conversation
        """
        return {
            "repo_path": str(self.daemon.repo_path),
            "current_branch": self.daemon.memory.session_data.get("current_branch"),
            "files_tracked": len(self.daemon.memory.file_contexts),
            "recent_insights": [i["insight"] for i in self.daemon.memory.get_recent_insights(3)],
            "recent_commits": [
                f"{c.get('short_hash', c['hash'][:7])}: {c['message']}"
                for c in self.daemon.memory.get_recent_commits(3)
            ],
        }


async def start_chat_session(daemon):
    """
    Start an interactive chat session with Lifeline

    Args:
        daemon: The Lifeline daemon instance
    """
    session = InteractiveChatSession(daemon)
    await session.start()
