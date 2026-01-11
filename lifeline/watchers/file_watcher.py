"""
File Watcher - The Eyes of Lifeline

Monitors file system changes in real-time, tracking every modification,
creation, and deletion in the codebase.
"""

import asyncio
import logging
from pathlib import Path
from typing import Set, Optional, Dict, Any
from datetime import datetime

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("watchdog not available, using polling fallback")


logger = logging.getLogger(__name__)


class LifelineFileHandler(FileSystemEventHandler):
    """
    File system event handler for watchdog
    """

    def __init__(self, event_loop, repo_path: Path):
        self.event_loop = event_loop
        self.repo_path = repo_path
        self.ignored_patterns = {
            ".git",
            "__pycache__",
            "*.pyc",
            ".lifeline",
            "node_modules",
            ".pytest_cache",
            ".venv",
            "venv",
        }

    def _should_ignore(self, path: str) -> bool:
        """Check if path should be ignored"""
        path_obj = Path(path)

        # Check against ignored patterns
        for pattern in self.ignored_patterns:
            if pattern.startswith("*."):
                # File extension pattern
                if path_obj.suffix == pattern[1:]:
                    return True
            else:
                # Directory or exact match
                if pattern in path_obj.parts:
                    return True

        return False

    def on_modified(self, event: FileSystemEvent):
        if event.is_directory or self._should_ignore(event.src_path):
            return

        asyncio.create_task(self.event_loop.emit("file:changed", {
            "path": event.src_path,
            "type": "modified",
            "timestamp": datetime.now().isoformat(),
        }))

    def on_created(self, event: FileSystemEvent):
        if event.is_directory or self._should_ignore(event.src_path):
            return

        asyncio.create_task(self.event_loop.emit("file:created", {
            "path": event.src_path,
            "type": "created",
            "timestamp": datetime.now().isoformat(),
        }))

    def on_deleted(self, event: FileSystemEvent):
        if event.is_directory or self._should_ignore(event.src_path):
            return

        asyncio.create_task(self.event_loop.emit("file:deleted", {
            "path": event.src_path,
            "type": "deleted",
            "timestamp": datetime.now().isoformat(),
        }))


class FileWatcher:
    """
    Watches file system for changes

    This is one of the primary senses of Lifeline - it sees every
    change to the codebase as it happens.
    """

    def __init__(self, repo_path: Path, event_loop):
        self.repo_path = repo_path
        self.event_loop = event_loop
        self.observer: Optional[Observer] = None
        self.is_watching = False
        self.watch_count = 0

        # Fallback polling if watchdog not available
        self.use_polling = not WATCHDOG_AVAILABLE
        self.polling_task = None
        self.last_scan: Dict[str, float] = {}

    async def start(self):
        """
        Start watching the file system - open your eyes!
        """
        logger.info("👀 File watcher awakening...")

        if self.use_polling:
            await self._start_polling()
        else:
            await self._start_watchdog()

        self.is_watching = True
        logger.info(f"✅ File watcher active, monitoring: {self.repo_path}")

    async def _start_watchdog(self):
        """Start watchdog-based monitoring"""
        handler = LifelineFileHandler(self.event_loop, self.repo_path)
        self.observer = Observer()
        self.observer.schedule(handler, str(self.repo_path), recursive=True)
        self.observer.start()

        logger.info("Using watchdog for file monitoring")

    async def _start_polling(self):
        """Start polling-based monitoring (fallback)"""
        logger.info("Using polling for file monitoring")
        self.polling_task = asyncio.create_task(self._poll_files())

    async def _poll_files(self):
        """
        Poll file system for changes (fallback when watchdog unavailable)
        """
        while self.is_watching:
            try:
                await self._scan_directory()
                await asyncio.sleep(2)  # Poll every 2 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error polling files: {e}")

    async def _scan_directory(self):
        """Scan directory for changes"""
        current_files = {}

        # Scan Python files
        for file_path in self.repo_path.rglob("*.py"):
            if self._should_ignore(file_path):
                continue

            try:
                stat = file_path.stat()
                mtime = stat.st_mtime
                current_files[str(file_path)] = mtime

                # Check if file is new or modified
                if str(file_path) not in self.last_scan:
                    await self.event_loop.emit("file:created", {
                        "path": str(file_path),
                        "type": "created",
                    })
                elif self.last_scan[str(file_path)] != mtime:
                    await self.event_loop.emit("file:changed", {
                        "path": str(file_path),
                        "type": "modified",
                    })

            except Exception as e:
                logger.debug(f"Error scanning {file_path}: {e}")

        # Check for deleted files
        for old_file in self.last_scan:
            if old_file not in current_files:
                await self.event_loop.emit("file:deleted", {
                    "path": old_file,
                    "type": "deleted",
                })

        self.last_scan = current_files

    def _should_ignore(self, path: Path) -> bool:
        """Check if path should be ignored"""
        ignored = {
            ".git",
            "__pycache__",
            ".lifeline",
            "node_modules",
            ".pytest_cache",
            ".venv",
            "venv",
        }

        return any(part in ignored for part in path.parts)

    async def stop(self):
        """
        Stop watching - close your eyes
        """
        logger.info("👀 File watcher going to sleep...")

        self.is_watching = False

        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=5)

        if self.polling_task:
            self.polling_task.cancel()
            try:
                await self.polling_task
            except asyncio.CancelledError:
                pass

        logger.info("✅ File watcher stopped")

    def get_watch_count(self) -> int:
        """Get number of files being watched"""
        return self.watch_count
