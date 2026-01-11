"""
AI Decision Engine - The Conscious Mind

This is the brain of Lifeline - it analyzes events, makes decisions,
and determines when and how to proactively assist developers.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

from lifeline.ai.transformers_brain import TransformersBrain


logger = logging.getLogger(__name__)


class AIDecisionEngine:
    """
    The AI brain that makes Lifeline intelligent

    Uses transformer models to understand code changes, detect patterns,
    identify issues, and make proactive suggestions.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.brain = TransformersBrain(config)

        # Decision thresholds
        self.alert_threshold = config.get("alert_threshold", 0.7)
        self.suggestion_threshold = config.get("suggestion_threshold", 0.5)

        # Track insights
        self.insights_generated = 0
        self.suggestions_made = 0
        self.alerts_raised = 0

        # Learning memory
        self.pattern_memory: List[Dict[str, Any]] = []
        self.issue_memory: List[Dict[str, Any]] = []

    async def initialize(self):
        """
        Initialize the AI brain
        """
        logger.info("🧠 AI Decision Engine awakening...")

        await self.brain.initialize()

        logger.info("✅ AI Decision Engine ready")
        logger.info("💭 Consciousness: ACTIVE")
        logger.info("🎯 Ready to learn and assist")

    async def analyze_file_change(self, file_path: str, event: Dict[str, Any]):
        """
        Analyze a file change and decide if action is needed

        Args:
            file_path: Path to changed file
            event: File change event data
        """
        logger.debug(f"🧠 Analyzing file change: {file_path}")

        try:
            # Read the file content
            content = await self._read_file(file_path)

            if not content:
                return

            # Analyze with AI brain
            analysis = await self.brain.analyze_code(content, file_path)

            # Make decisions based on analysis
            await self._process_analysis(analysis, "file_change", {
                "file_path": file_path,
                "event": event,
            })

        except Exception as e:
            logger.error(f"Error analyzing file change: {e}")

    async def analyze_new_file(self, file_path: str):
        """
        Analyze a newly created file

        Args:
            file_path: Path to new file
        """
        logger.debug(f"🧠 Analyzing new file: {file_path}")

        try:
            content = await self._read_file(file_path)

            if not content:
                return

            analysis = await self.brain.analyze_code(content, file_path)

            # Check if this file introduces potential issues
            if analysis.get("potential_issues"):
                await self._raise_alert(
                    "New file may have issues",
                    {
                        "file": file_path,
                        "issues": analysis["potential_issues"],
                    },
                    priority="normal"
                )

        except Exception as e:
            logger.error(f"Error analyzing new file: {e}")

    async def analyze_commit(self, commit_data: Dict[str, Any]):
        """
        Analyze a git commit

        Args:
            commit_data: Commit information
        """
        logger.debug(f"🧠 Analyzing commit: {commit_data.get('short_hash')}")

        try:
            message = commit_data.get("message", "")
            files_changed = commit_data.get("files_changed", [])

            # Analyze commit message quality
            if len(message) < 10:
                await self._make_suggestion(
                    "Consider more descriptive commit messages",
                    {
                        "commit": commit_data.get("short_hash"),
                        "current_message": message,
                    }
                )

            # Analyze scope of changes
            if len(files_changed) > 20:
                await self._make_suggestion(
                    f"Large commit detected ({len(files_changed)} files). Consider breaking into smaller commits.",
                    {
                        "commit": commit_data.get("short_hash"),
                        "files_count": len(files_changed),
                    }
                )

            # Learn from commit patterns
            await self._learn_commit_pattern(commit_data)

        except Exception as e:
            logger.error(f"Error analyzing commit: {e}")

    async def analyze_merge(self, merge_data: Dict[str, Any]):
        """
        Analyze a merge operation

        Args:
            merge_data: Merge information
        """
        logger.debug("🧠 Analyzing merge...")

        # Merges are critical operations - always worth noting
        await self._generate_insight(
            "Merge operation detected - reviewing integration",
            merge_data
        )

    async def _process_analysis(self, analysis: Dict[str, Any], context_type: str, context: Dict[str, Any]):
        """
        Process AI analysis and make decisions
        """
        # Check for issues
        if "issues" in analysis:
            for issue in analysis["issues"]:
                severity = issue.get("severity", "normal")

                if severity == "high":
                    await self._raise_alert(
                        issue.get("description", "Issue detected"),
                        {**context, "issue": issue},
                        priority="high"
                    )
                elif severity == "medium":
                    await self._make_suggestion(
                        issue.get("description", "Potential improvement"),
                        {**context, "issue": issue}
                    )

        # Check for patterns
        if "patterns" in analysis:
            await self._learn_patterns(analysis["patterns"])

        # Generate insights
        if analysis.get("confidence", 0) > self.suggestion_threshold:
            if "insight" in analysis:
                await self._generate_insight(
                    analysis["insight"],
                    context
                )

    async def _generate_insight(self, insight: str, context: Dict[str, Any]):
        """
        Generate and emit an AI insight
        """
        self.insights_generated += 1

        logger.info(f"💡 Insight: {insight}")

        # Could emit event here to notify user
        # await self.event_loop.emit("ai:insight", {"insight": insight, "context": context})

    async def _make_suggestion(self, suggestion: str, context: Dict[str, Any]):
        """
        Make a proactive suggestion
        """
        self.suggestions_made += 1

        logger.info(f"💭 Suggestion: {suggestion}")

        # Could emit event here
        # await self.event_loop.emit("ai:suggestion", {"suggestion": suggestion, "context": context})

    async def _raise_alert(self, alert: str, context: Dict[str, Any], priority: str = "normal"):
        """
        Raise an alert for important issues
        """
        self.alerts_raised += 1

        logger.warning(f"⚠️  Alert [{priority}]: {alert}")

        # Could emit event here
        # await self.event_loop.emit("ai:alert", {"alert": alert, "priority": priority, "context": context})

    async def _learn_patterns(self, patterns: List[Dict[str, Any]]):
        """
        Learn from code patterns
        """
        for pattern in patterns:
            self.pattern_memory.append({
                "pattern": pattern,
                "timestamp": datetime.now(),
            })

        # Keep memory bounded
        if len(self.pattern_memory) > 1000:
            self.pattern_memory = self.pattern_memory[-1000:]

    async def _learn_commit_pattern(self, commit_data: Dict[str, Any]):
        """
        Learn from commit patterns
        """
        # Store commit pattern for learning
        pattern = {
            "message_length": len(commit_data.get("message", "")),
            "files_count": len(commit_data.get("files_changed", [])),
            "timestamp": commit_data.get("timestamp"),
        }

        self.pattern_memory.append(pattern)

    async def _read_file(self, file_path: str) -> Optional[str]:
        """
        Read file contents
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return None

            # Only read text files
            if path.suffix not in [".py", ".txt", ".md", ".json", ".yaml", ".yml"]:
                return None

            # Don't read huge files
            if path.stat().st_size > 1_000_000:  # 1MB limit
                logger.debug(f"Skipping large file: {file_path}")
                return None

            return path.read_text(encoding="utf-8", errors="ignore")

        except Exception as e:
            logger.debug(f"Could not read file {file_path}: {e}")
            return None

    def get_stats(self) -> Dict[str, int]:
        """
        Get AI engine statistics
        """
        return {
            "insights_generated": self.insights_generated,
            "suggestions_made": self.suggestions_made,
            "alerts_raised": self.alerts_raised,
            "patterns_learned": len(self.pattern_memory),
        }
