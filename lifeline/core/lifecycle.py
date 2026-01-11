"""
Lifecycle Manager - Birth, Life, and Rest

Manages the lifecycle of the daemon and its components.
"""

import logging
import asyncio
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lifeline.core.daemon import LifelineDaemon


logger = logging.getLogger(__name__)


class LifecycleManager:
    """
    Manages the lifecycle of the Lifeline daemon

    Handles initialization, health checks, and graceful shutdown.
    """

    def __init__(self, daemon: "LifelineDaemon"):
        self.daemon = daemon
        self.startup_time = None
        self.health_check_task = None
        self.health_check_interval = 60  # seconds

    async def startup(self):
        """
        Initialize all systems - the moment of awakening
        """
        logger.info("🌅 Beginning startup sequence...")

        self.startup_time = datetime.now()

        # Perform health checks
        await self._initial_health_check()

        # Start periodic health monitoring
        self.health_check_task = asyncio.create_task(self._health_monitor())

        logger.info("✅ Startup sequence complete")

    async def shutdown(self):
        """
        Graceful shutdown - time to rest
        """
        logger.info("🌙 Beginning shutdown sequence...")

        # Cancel health check task
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass

        # Cleanup resources
        await self._cleanup()

        logger.info("✅ Shutdown sequence complete")

    async def _initial_health_check(self):
        """
        Perform initial health checks before going live
        """
        logger.info("🏥 Performing initial health check...")

        checks = {
            "repository_exists": self.daemon.repo_path.exists(),
            "is_git_repo": (self.daemon.repo_path / ".git").exists(),
            "memory_writable": True,  # Will check actual write permissions
        }

        # Check if we can write to memory location
        try:
            memory_path = self.daemon.repo_path / ".lifeline"
            memory_path.mkdir(exist_ok=True)
            checks["memory_writable"] = True
        except Exception as e:
            logger.warning(f"Memory location not writable: {e}")
            checks["memory_writable"] = False

        # Log results
        for check, status in checks.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"  {status_icon} {check}: {status}")

        if not all(checks.values()):
            logger.warning("⚠️  Some health checks failed, but continuing anyway")

    async def _health_monitor(self):
        """
        Continuously monitor system health
        """
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")

    async def _perform_health_check(self):
        """
        Perform periodic health check
        """
        logger.debug("🏥 Performing health check...")

        # Check event loop responsiveness
        event_count = self.daemon.event_loop.get_event_count()

        # Check memory status
        memory_size = self.daemon.memory.get_size()

        logger.debug(f"  Events processed: {event_count}")
        logger.debug(f"  Memory size: {memory_size} bytes")

    async def _cleanup(self):
        """
        Clean up resources before shutdown
        """
        logger.info("🧹 Cleaning up resources...")

        # Could add cleanup tasks here:
        # - Flush logs
        # - Clear temporary files
        # - Send final notifications

        logger.info("✅ Cleanup complete")

    def get_uptime(self) -> float:
        """
        Get daemon uptime in seconds
        """
        if not self.startup_time:
            return 0.0

        delta = datetime.now() - self.startup_time
        return delta.total_seconds()
