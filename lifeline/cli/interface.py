"""
Lifeline CLI - Command-line Interface

Control and interact with the Lifeline daemon.
"""

import sys
import asyncio
import argparse
import json
from pathlib import Path
from typing import Optional

from lifeline.core.daemon import LifelineDaemon


class LifelineCLI:
    """
    Command-line interface for Lifeline

    Provides commands to start, stop, query, and interact with the daemon.
    """

    def __init__(self):
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """
        Create argument parser
        """
        parser = argparse.ArgumentParser(
            prog="lifeline",
            description="✨ Lifeline - The Living AI Daemon for Transformers",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  lifeline run                    Start the daemon
  lifeline status                 Get daemon status
  lifeline config --show          Show configuration
  lifeline memory --stats         Show memory statistics

For more information, visit: https://github.com/huggingface/transformers
            """
        )

        parser.add_argument(
            "--repo",
            type=Path,
            default=Path.cwd(),
            help="Repository path (default: current directory)"
        )

        parser.add_argument(
            "--config",
            type=Path,
            help="Configuration file path"
        )

        subparsers = parser.add_subparsers(dest="command", help="Command to run")

        # Run command
        run_parser = subparsers.add_parser(
            "run",
            help="Start the Lifeline daemon"
        )
        run_parser.add_argument(
            "--foreground",
            action="store_true",
            help="Run in foreground (default)"
        )

        # Status command
        status_parser = subparsers.add_parser(
            "status",
            help="Get daemon status"
        )
        status_parser.add_argument(
            "--json",
            action="store_true",
            help="Output as JSON"
        )

        # Memory command
        memory_parser = subparsers.add_parser(
            "memory",
            help="Memory operations"
        )
        memory_parser.add_argument(
            "--stats",
            action="store_true",
            help="Show memory statistics"
        )
        memory_parser.add_argument(
            "--insights",
            type=int,
            metavar="N",
            help="Show last N insights"
        )
        memory_parser.add_argument(
            "--commits",
            type=int,
            metavar="N",
            help="Show last N commits"
        )

        # Config command
        config_parser = subparsers.add_parser(
            "config",
            help="Configuration operations"
        )
        config_parser.add_argument(
            "--show",
            action="store_true",
            help="Show current configuration"
        )
        config_parser.add_argument(
            "--init",
            action="store_true",
            help="Initialize default configuration"
        )

        return parser

    def run(self, argv: Optional[list] = None):
        """
        Run CLI with arguments

        Args:
            argv: Command-line arguments (default: sys.argv[1:])
        """
        args = self.parser.parse_args(argv)

        if not args.command:
            self.parser.print_help()
            sys.exit(1)

        try:
            if args.command == "run":
                asyncio.run(self._cmd_run(args))
            elif args.command == "status":
                asyncio.run(self._cmd_status(args))
            elif args.command == "memory":
                asyncio.run(self._cmd_memory(args))
            elif args.command == "config":
                self._cmd_config(args)
            else:
                self.parser.print_help()
                sys.exit(1)

        except KeyboardInterrupt:
            print("\n✨ Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    async def _cmd_run(self, args):
        """
        Run the daemon
        """
        print("✨ Starting Lifeline daemon...")
        print(f"📍 Repository: {args.repo}")
        print()

        config = self._load_config(args.config) if args.config else {}

        daemon = LifelineDaemon(repo_path=args.repo, config=config)

        try:
            await daemon.start()
        except KeyboardInterrupt:
            print("\n🌙 Shutting down gracefully...")
            await daemon.stop()

    async def _cmd_status(self, args):
        """
        Show daemon status
        """
        daemon = LifelineDaemon(repo_path=args.repo)
        status = daemon.get_status()

        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print("📊 Lifeline Status")
            print("=" * 50)
            print(f"Alive: {status['alive']}")
            print(f"Repository: {status['repo_path']}")
            print(f"Birth time: {status['birth_time']}")
            print(f"Uptime: {status['uptime']}")
            print(f"Memory size: {status['memory_size']} bytes")
            print(f"Events processed: {status['events_processed']}")
            print(f"Files watching: {status['watching_files']}")

    async def _cmd_memory(self, args):
        """
        Memory operations
        """
        from lifeline.memory.context_manager import ContextManager

        memory = ContextManager(args.repo)
        await memory.load()

        if args.stats:
            stats = memory.get_stats()
            print("🧠 Memory Statistics")
            print("=" * 50)
            print(f"Total events seen: {stats['total_events_seen']}")
            print(f"Files tracked: {stats['files_tracked']}")
            print(f"Insights stored: {stats['insights_stored']}")
            print(f"Commits remembered: {stats['commits_remembered']}")
            print(f"Patterns learned: {stats['patterns_learned']}")
            print(f"Memory size: {stats['memory_size_bytes']} bytes")
            print(f"Created at: {stats['created_at']}")
            print(f"Last saved: {stats['last_saved']}")

        if args.insights:
            insights = memory.get_recent_insights(args.insights)
            print(f"\n💡 Recent Insights (last {args.insights})")
            print("=" * 50)
            for i, insight in enumerate(insights, 1):
                print(f"\n{i}. {insight['insight']}")
                print(f"   Time: {insight['timestamp']}")

        if args.commits:
            commits = memory.get_recent_commits(args.commits)
            print(f"\n📦 Recent Commits (last {args.commits})")
            print("=" * 50)
            for i, commit in enumerate(commits, 1):
                print(f"\n{i}. {commit.get('short_hash', commit['hash'][:7])}: {commit['message']}")
                print(f"   Author: {commit['author']}")
                print(f"   Time: {commit['timestamp']}")
                print(f"   Files: {len(commit['files_changed'])}")

    def _cmd_config(self, args):
        """
        Configuration operations
        """
        config_file = args.repo / ".lifeline" / "config.json"

        if args.show:
            if config_file.exists():
                config = json.loads(config_file.read_text())
                print("⚙️  Current Configuration")
                print("=" * 50)
                print(json.dumps(config, indent=2))
            else:
                print("No configuration file found")
                print(f"Expected location: {config_file}")

        if args.init:
            default_config = {
                "log_level": "INFO",
                "ai": {
                    "model_name": "gpt2",
                    "use_local": True,
                    "alert_threshold": 0.7,
                    "suggestion_threshold": 0.5,
                },
                "watchers": {
                    "file_poll_interval": 2.0,
                    "git_poll_interval": 5.0,
                },
            }

            config_file.parent.mkdir(exist_ok=True)
            config_file.write_text(json.dumps(default_config, indent=2))

            print("✅ Configuration initialized")
            print(f"📝 Location: {config_file}")

    def _load_config(self, config_path: Path) -> dict:
        """
        Load configuration from file
        """
        if not config_path.exists():
            return {}

        return json.loads(config_path.read_text())


def main():
    """
    Main entry point
    """
    cli = LifelineCLI()
    cli.run()


if __name__ == "__main__":
    main()
