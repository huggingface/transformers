"""
The Lifeline Daemon - The Heart of the Living AI System

This daemon runs continuously, maintaining awareness and proactively
assisting with the transformers codebase.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from lifeline.core.event_loop import EventLoop
from lifeline.core.lifecycle import LifecycleManager
from lifeline.watchers.file_watcher import FileWatcher
from lifeline.watchers.git_watcher import GitWatcher
from lifeline.ai.decision_engine import AIDecisionEngine
from lifeline.memory.context_manager import ContextManager


logger = logging.getLogger(__name__)


class LifelineDaemon:
    """
    The Living AI Daemon

    Continuously monitors the codebase, maintains context, and proactively
    assists developers. This is not just a tool - it's a companion.
    """

    def __init__(self, repo_path: Optional[Path] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Lifeline daemon.

        Args:
            repo_path: Path to the repository to monitor
            config: Configuration dictionary
        """
        self.repo_path = repo_path or Path.cwd()
        self.config = config or {}
        self.is_alive = False
        self.birth_time = None

        # Core components
        self.lifecycle = LifecycleManager(self)
        self.event_loop = EventLoop()
        self.file_watcher = FileWatcher(self.repo_path, self.event_loop)
        self.git_watcher = GitWatcher(self.repo_path, self.event_loop)
        self.ai_engine = AIDecisionEngine(self.config.get("ai", {}))
        self.memory = ContextManager(self.repo_path)

        # Optional components
        self.web_dashboard = None
        self.voice = None

        # Enable web dashboard if configured
        if self.config.get("web", {}).get("enabled", False):
            try:
                from lifeline.web.dashboard import LifelineDashboard
                port = self.config.get("web", {}).get("port", 8765)
                self.web_dashboard = LifelineDashboard(self, port=port)
                logger.info(f"🌐 Web dashboard will run on port {port}")
            except ImportError:
                logger.warning("aiohttp not available - web dashboard disabled")

        # Enable conversation if configured
        if self.config.get("conversation", {}).get("enabled", False):
            try:
                from lifeline.conversation.voice import TransformerVoice
                self.voice = TransformerVoice(self.config.get("conversation", {}))
                logger.info("💬 Conversation features enabled")
            except ImportError:
                logger.warning("transformers not available - conversation disabled")

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("✨ Lifeline daemon initialized")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(self.stop())

    async def start(self):
        """
        Awaken the daemon - bring it to life!
        """
        logger.info("🌟 Awakening Lifeline...")

        self.is_alive = True
        self.birth_time = datetime.now()

        # Initialize all components
        await self.lifecycle.startup()
        await self.memory.load()
        await self.file_watcher.start()
        await self.git_watcher.start()
        await self.ai_engine.initialize()

        # Start optional components
        if self.web_dashboard:
            await self.web_dashboard.start()

        if self.voice:
            await self.voice.initialize()

        logger.info(f"✨ Lifeline is now ALIVE at {self.birth_time}")
        logger.info(f"📍 Watching: {self.repo_path}")
        logger.info("🧠 Awareness: ACTIVE")
        logger.info("💚 Status: Ready to assist")

        if self.web_dashboard:
            logger.info(f"🌐 Dashboard: http://localhost:{self.web_dashboard.port}")
        if self.voice and self.voice.is_ready:
            logger.info("💬 Voice: Ready for conversation")

        # Register event handlers
        self._register_event_handlers()

        # Start the main event loop
        await self.event_loop.run()

    async def stop(self):
        """
        Gracefully stop the daemon
        """
        if not self.is_alive:
            return

        logger.info("💫 Initiating graceful shutdown...")

        self.is_alive = False

        # Stop all components
        await self.file_watcher.stop()
        await self.git_watcher.stop()
        await self.memory.save()

        # Stop optional components
        if self.web_dashboard:
            await self.web_dashboard.stop()

        if self.voice:
            await self.voice.shutdown()

        await self.lifecycle.shutdown()

        uptime = datetime.now() - self.birth_time if self.birth_time else None
        logger.info(f"✨ Lifeline has rested. Uptime: {uptime}")
        logger.info("👋 Until next time...")

    def _register_event_handlers(self):
        """
        Register handlers for various events
        """
        # File change events
        self.event_loop.on("file:changed", self._on_file_changed)
        self.event_loop.on("file:created", self._on_file_created)
        self.event_loop.on("file:deleted", self._on_file_deleted)

        # Git events
        self.event_loop.on("git:commit", self._on_git_commit)
        self.event_loop.on("git:branch_change", self._on_branch_change)
        self.event_loop.on("git:merge", self._on_git_merge)

        # AI decision points
        self.event_loop.on("ai:insight", self._on_ai_insight)
        self.event_loop.on("ai:suggestion", self._on_ai_suggestion)
        self.event_loop.on("ai:alert", self._on_ai_alert)

    async def _on_file_changed(self, event):
        """Handle file change events"""
        file_path = event["path"]
        logger.info(f"📝 File changed: {file_path}")

        # Update context
        await self.memory.update_file_context(file_path)

        # Let AI analyze the change
        await self.ai_engine.analyze_file_change(file_path, event)

    async def _on_file_created(self, event):
        """Handle file creation events"""
        file_path = event["path"]
        logger.info(f"✨ New file created: {file_path}")

        await self.memory.add_file(file_path)
        await self.ai_engine.analyze_new_file(file_path)

    async def _on_file_deleted(self, event):
        """Handle file deletion events"""
        file_path = event["path"]
        logger.warning(f"🗑️  File deleted: {file_path}")

        await self.memory.remove_file(file_path)

    async def _on_git_commit(self, event):
        """Handle git commit events"""
        commit_hash = event.get("hash", "unknown")
        message = event.get("message", "")
        logger.info(f"📦 New commit: {commit_hash[:7]} - {message}")

        # Update memory with commit context
        await self.memory.record_commit(event)

        # Analyze commit for insights
        await self.ai_engine.analyze_commit(event)

    async def _on_branch_change(self, event):
        """Handle branch change events"""
        from_branch = event.get("from")
        to_branch = event.get("to")
        logger.info(f"🌿 Branch changed: {from_branch} → {to_branch}")

        await self.memory.update_branch_context(to_branch)

    async def _on_git_merge(self, event):
        """Handle merge events"""
        logger.info(f"🔀 Merge detected: {event}")
        await self.ai_engine.analyze_merge(event)

    async def _on_ai_insight(self, event):
        """Handle AI insights"""
        insight = event.get("insight", "")
        logger.info(f"💡 AI Insight: {insight}")

        # Store insight in memory
        await self.memory.record_insight(insight)

    async def _on_ai_suggestion(self, event):
        """Handle AI suggestions"""
        suggestion = event.get("suggestion", "")
        logger.info(f"💭 AI Suggestion: {suggestion}")

    async def _on_ai_alert(self, event):
        """Handle AI alerts"""
        alert = event.get("alert", "")
        priority = event.get("priority", "normal")
        logger.warning(f"⚠️  AI Alert [{priority}]: {alert}")

    def get_status(self) -> Dict[str, Any]:
        """
        Get current daemon status
        """
        uptime = datetime.now() - self.birth_time if self.birth_time else None

        return {
            "alive": self.is_alive,
            "birth_time": self.birth_time.isoformat() if self.birth_time else None,
            "uptime": str(uptime) if uptime else None,
            "repo_path": str(self.repo_path),
            "memory_size": self.memory.get_size(),
            "events_processed": self.event_loop.get_event_count(),
            "watching_files": self.file_watcher.get_watch_count(),
        }


async def main():
    """Main entry point for running the daemon"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    daemon = LifelineDaemon()

    try:
        await daemon.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        await daemon.stop()


if __name__ == "__main__":
    asyncio.run(main())
