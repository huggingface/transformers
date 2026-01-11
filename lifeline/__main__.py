"""
Main entry point for running Lifeline

Usage:
    python -m lifeline run
    python -m lifeline status
    python -m lifeline memory --stats
"""

from lifeline.cli.interface import main

if __name__ == "__main__":
    main()
