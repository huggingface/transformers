"""
Context Manager - The Long-term Memory

Persists knowledge, learns from patterns, and maintains context
across sessions. This is what allows Lifeline to truly remember.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio


logger = logging.getLogger(__name__)


class ContextManager:
    """
    Manages persistent memory and context

    Stores insights, patterns, file states, and learning across sessions.
    This is the long-term memory that makes Lifeline truly intelligent.
    """

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.memory_dir = repo_path / ".lifeline"
        self.memory_file = self.memory_dir / "memory.json"

        # In-memory state
        self.file_contexts: Dict[str, Dict[str, Any]] = {}
        self.insights: List[Dict[str, Any]] = []
        self.commit_history: List[Dict[str, Any]] = []
        self.learned_patterns: List[Dict[str, Any]] = []
        self.session_data: Dict[str, Any] = {}

        # Metadata
        self.created_at: Optional[datetime] = None
        self.last_saved: Optional[datetime] = None
        self.total_events_seen = 0

    async def load(self):
        """
        Load persistent memory from disk
        """
        logger.info("🧠 Loading memory...")

        try:
            self.memory_dir.mkdir(exist_ok=True)

            if self.memory_file.exists():
                data = json.loads(self.memory_file.read_text())

                self.file_contexts = data.get("file_contexts", {})
                self.insights = data.get("insights", [])
                self.commit_history = data.get("commit_history", [])
                self.learned_patterns = data.get("learned_patterns", [])
                self.session_data = data.get("session_data", {})
                self.total_events_seen = data.get("total_events_seen", 0)

                if data.get("created_at"):
                    self.created_at = datetime.fromisoformat(data["created_at"])

                logger.info(f"✅ Memory loaded: {len(self.file_contexts)} files, {len(self.insights)} insights")
            else:
                self.created_at = datetime.now()
                logger.info("🌟 Starting fresh memory")

        except Exception as e:
            logger.error(f"Error loading memory: {e}")
            logger.info("Starting with empty memory")

    async def save(self):
        """
        Persist memory to disk
        """
        logger.info("💾 Saving memory...")

        try:
            self.memory_dir.mkdir(exist_ok=True)

            data = {
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "last_saved": datetime.now().isoformat(),
                "total_events_seen": self.total_events_seen,
                "file_contexts": self.file_contexts,
                "insights": self.insights[-1000:],  # Keep last 1000 insights
                "commit_history": self.commit_history[-500:],  # Keep last 500 commits
                "learned_patterns": self.learned_patterns[-1000:],  # Keep last 1000 patterns
                "session_data": self.session_data,
            }

            self.memory_file.write_text(json.dumps(data, indent=2))
            self.last_saved = datetime.now()

            logger.info("✅ Memory saved")

        except Exception as e:
            logger.error(f"Error saving memory: {e}")

    async def update_file_context(self, file_path: str):
        """
        Update context for a file

        Args:
            file_path: Path to file
        """
        self.total_events_seen += 1

        if file_path not in self.file_contexts:
            self.file_contexts[file_path] = {
                "first_seen": datetime.now().isoformat(),
                "modification_count": 0,
            }

        self.file_contexts[file_path]["last_modified"] = datetime.now().isoformat()
        self.file_contexts[file_path]["modification_count"] += 1

        logger.debug(f"Updated context for {file_path}")

    async def add_file(self, file_path: str):
        """
        Add a new file to memory

        Args:
            file_path: Path to new file
        """
        self.total_events_seen += 1

        self.file_contexts[file_path] = {
            "created": datetime.now().isoformat(),
            "first_seen": datetime.now().isoformat(),
            "modification_count": 0,
            "is_new": True,
        }

        logger.debug(f"Added new file to memory: {file_path}")

    async def remove_file(self, file_path: str):
        """
        Remove a file from memory

        Args:
            file_path: Path to deleted file
        """
        self.total_events_seen += 1

        if file_path in self.file_contexts:
            # Don't delete, just mark as deleted
            self.file_contexts[file_path]["deleted"] = datetime.now().isoformat()

        logger.debug(f"Marked file as deleted: {file_path}")

    async def record_commit(self, commit_data: Dict[str, Any]):
        """
        Record a commit in memory

        Args:
            commit_data: Commit information
        """
        self.total_events_seen += 1

        commit_record = {
            "hash": commit_data.get("hash"),
            "message": commit_data.get("message"),
            "author": commit_data.get("author"),
            "timestamp": commit_data.get("timestamp"),
            "files_changed": commit_data.get("files_changed", []),
            "recorded_at": datetime.now().isoformat(),
        }

        self.commit_history.append(commit_record)

        # Keep memory bounded
        if len(self.commit_history) > 1000:
            self.commit_history = self.commit_history[-1000:]

        logger.debug(f"Recorded commit: {commit_data.get('short_hash')}")

    async def record_insight(self, insight: str, context: Optional[Dict[str, Any]] = None):
        """
        Record an AI insight

        Args:
            insight: Insight text
            context: Optional context data
        """
        self.total_events_seen += 1

        insight_record = {
            "insight": insight,
            "context": context or {},
            "timestamp": datetime.now().isoformat(),
        }

        self.insights.append(insight_record)

        # Keep memory bounded
        if len(self.insights) > 2000:
            self.insights = self.insights[-2000:]

        logger.debug("Recorded insight")

    async def update_branch_context(self, branch: str):
        """
        Update current branch context

        Args:
            branch: Branch name
        """
        self.session_data["current_branch"] = branch
        self.session_data["branch_changed_at"] = datetime.now().isoformat()

        logger.debug(f"Updated branch context: {branch}")

    def get_file_context(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get context for a file

        Args:
            file_path: Path to file

        Returns:
            File context or None
        """
        return self.file_contexts.get(file_path)

    def get_recent_insights(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent insights

        Args:
            count: Number of insights to return

        Returns:
            List of recent insights
        """
        return self.insights[-count:]

    def get_recent_commits(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent commits

        Args:
            count: Number of commits to return

        Returns:
            List of recent commits
        """
        return self.commit_history[-count:]

    def get_size(self) -> int:
        """
        Get approximate size of memory in bytes

        Returns:
            Memory size in bytes
        """
        if self.memory_file.exists():
            return self.memory_file.stat().st_size
        return 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics

        Returns:
            Statistics dictionary
        """
        return {
            "total_events_seen": self.total_events_seen,
            "files_tracked": len(self.file_contexts),
            "insights_stored": len(self.insights),
            "commits_remembered": len(self.commit_history),
            "patterns_learned": len(self.learned_patterns),
            "memory_size_bytes": self.get_size(),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_saved": self.last_saved.isoformat() if self.last_saved else None,
        }
