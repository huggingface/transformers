"""
Git Watcher - The Memory Tracker

Monitors git operations, tracking commits, branch changes, merges,
and the evolution of the codebase over time.
"""

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


logger = logging.getLogger(__name__)


class GitWatcher:
    """
    Watches git repository for changes

    Tracks the evolution of the codebase through git operations,
    understanding the flow of development.
    """

    def __init__(self, repo_path: Path, event_loop):
        self.repo_path = repo_path
        self.event_loop = event_loop
        self.is_watching = False
        self.polling_task = None
        self.poll_interval = 5  # seconds

        # Track state
        self.current_branch = None
        self.last_commit_hash = None
        self.commit_count = 0

    async def start(self):
        """
        Start watching git - remember everything!
        """
        logger.info("🌿 Git watcher awakening...")

        # Initialize current state
        await self._initialize_state()

        # Start polling for changes
        self.is_watching = True
        self.polling_task = asyncio.create_task(self._poll_git())

        logger.info(f"✅ Git watcher active on branch: {self.current_branch}")

    async def _initialize_state(self):
        """
        Get initial git state
        """
        try:
            # Get current branch
            self.current_branch = await self._get_current_branch()

            # Get latest commit
            self.last_commit_hash = await self._get_latest_commit_hash()

            logger.info(f"📍 Current branch: {self.current_branch}")
            logger.info(f"📍 Latest commit: {self.last_commit_hash[:7] if self.last_commit_hash else 'none'}")

        except Exception as e:
            logger.warning(f"Could not initialize git state: {e}")

    async def _poll_git(self):
        """
        Poll git for changes
        """
        while self.is_watching:
            try:
                await self._check_for_changes()
                await asyncio.sleep(self.poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error polling git: {e}")

    async def _check_for_changes(self):
        """
        Check for git changes
        """
        # Check for branch changes
        current_branch = await self._get_current_branch()
        if current_branch != self.current_branch:
            await self._on_branch_change(self.current_branch, current_branch)
            self.current_branch = current_branch

        # Check for new commits
        latest_commit = await self._get_latest_commit_hash()
        if latest_commit != self.last_commit_hash and latest_commit is not None:
            await self._on_new_commit(latest_commit)
            self.last_commit_hash = latest_commit

    async def _on_branch_change(self, from_branch: Optional[str], to_branch: Optional[str]):
        """
        Handle branch change event
        """
        logger.info(f"🌿 Branch changed: {from_branch} → {to_branch}")

        await self.event_loop.emit("git:branch_change", {
            "from": from_branch,
            "to": to_branch,
            "timestamp": datetime.now().isoformat(),
        })

    async def _on_new_commit(self, commit_hash: str):
        """
        Handle new commit event
        """
        self.commit_count += 1

        # Get commit details
        commit_info = await self._get_commit_info(commit_hash)

        logger.info(f"📦 New commit detected: {commit_hash[:7]}")

        await self.event_loop.emit("git:commit", {
            "hash": commit_hash,
            "short_hash": commit_hash[:7],
            "message": commit_info.get("message", ""),
            "author": commit_info.get("author", ""),
            "timestamp": commit_info.get("timestamp", ""),
            "files_changed": commit_info.get("files_changed", []),
        })

    async def _get_current_branch(self) -> Optional[str]:
        """
        Get current git branch
        """
        try:
            result = await self._run_git_command(["branch", "--show-current"])
            return result.strip() if result else None
        except Exception:
            return None

    async def _get_latest_commit_hash(self) -> Optional[str]:
        """
        Get latest commit hash
        """
        try:
            result = await self._run_git_command(["rev-parse", "HEAD"])
            return result.strip() if result else None
        except Exception:
            return None

    async def _get_commit_info(self, commit_hash: str) -> Dict[str, Any]:
        """
        Get detailed commit information
        """
        info = {}

        try:
            # Get commit message
            message = await self._run_git_command([
                "log", "-1", "--pretty=%B", commit_hash
            ])
            info["message"] = message.strip() if message else ""

            # Get author
            author = await self._run_git_command([
                "log", "-1", "--pretty=%an <%ae>", commit_hash
            ])
            info["author"] = author.strip() if author else ""

            # Get timestamp
            timestamp = await self._run_git_command([
                "log", "-1", "--pretty=%cI", commit_hash
            ])
            info["timestamp"] = timestamp.strip() if timestamp else ""

            # Get files changed
            files = await self._run_git_command([
                "diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash
            ])
            info["files_changed"] = [f.strip() for f in files.split("\n") if f.strip()]

        except Exception as e:
            logger.warning(f"Could not get full commit info: {e}")

        return info

    async def _run_git_command(self, args: list) -> str:
        """
        Run a git command and return output
        """
        try:
            process = await asyncio.create_subprocess_exec(
                "git",
                *args,
                cwd=self.repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error = stderr.decode().strip()
                raise Exception(f"Git command failed: {error}")

            return stdout.decode()

        except Exception as e:
            logger.debug(f"Git command error: {e}")
            raise

    async def stop(self):
        """
        Stop watching git
        """
        logger.info("🌿 Git watcher going to sleep...")

        self.is_watching = False

        if self.polling_task:
            self.polling_task.cancel()
            try:
                await self.polling_task
            except asyncio.CancelledError:
                pass

        logger.info("✅ Git watcher stopped")
