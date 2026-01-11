"""
Event Loop - The Nervous System of Lifeline

Handles all events and coordinates communication between components.
"""

import asyncio
import logging
from collections import defaultdict
from typing import Callable, Dict, List, Any
from datetime import datetime


logger = logging.getLogger(__name__)


class EventLoop:
    """
    Asynchronous event loop for the daemon

    This is the nervous system - it routes signals between all components,
    allowing them to react to changes in real-time.
    """

    def __init__(self):
        self.handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_queue = asyncio.Queue()
        self.is_running = False
        self.event_count = 0
        self.event_history: List[Dict[str, Any]] = []
        self.max_history = 1000

    def on(self, event_type: str, handler: Callable):
        """
        Register an event handler

        Args:
            event_type: Type of event to listen for (e.g., "file:changed")
            handler: Async function to call when event occurs
        """
        self.handlers[event_type].append(handler)
        logger.debug(f"Registered handler for {event_type}")

    def off(self, event_type: str, handler: Callable):
        """
        Unregister an event handler

        Args:
            event_type: Type of event
            handler: Handler to remove
        """
        if handler in self.handlers[event_type]:
            self.handlers[event_type].remove(handler)
            logger.debug(f"Unregistered handler for {event_type}")

    async def emit(self, event_type: str, data: Dict[str, Any] = None):
        """
        Emit an event

        Args:
            event_type: Type of event
            data: Event data
        """
        event = {
            "type": event_type,
            "data": data or {},
            "timestamp": datetime.now(),
        }

        await self.event_queue.put(event)
        logger.debug(f"Emitted event: {event_type}")

    async def run(self):
        """
        Start the event loop - the heartbeat of the system
        """
        self.is_running = True
        logger.info("💓 Event loop started - heartbeat active")

        while self.is_running:
            try:
                # Wait for next event with timeout
                event = await asyncio.wait_for(
                    self.event_queue.get(),
                    timeout=1.0
                )

                await self._process_event(event)

            except asyncio.TimeoutError:
                # No events, continue waiting
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}", exc_info=True)

        logger.info("💓 Event loop stopped")

    async def _process_event(self, event: Dict[str, Any]):
        """
        Process a single event by calling all registered handlers
        """
        event_type = event["type"]
        event_data = event["data"]

        self.event_count += 1

        # Store in history
        self._record_event(event)

        # Get handlers for this event type
        handlers = self.handlers.get(event_type, [])

        if not handlers:
            logger.debug(f"No handlers registered for {event_type}")
            return

        # Execute all handlers concurrently
        tasks = [handler(event_data) for handler in handlers]

        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error in event handlers for {event_type}: {e}")

    def _record_event(self, event: Dict[str, Any]):
        """Record event in history for learning"""
        self.event_history.append(event)

        # Keep history bounded
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]

    async def stop(self):
        """Stop the event loop"""
        self.is_running = False

    def get_event_count(self) -> int:
        """Get total number of events processed"""
        return self.event_count

    def get_recent_events(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent events from history"""
        return self.event_history[-count:]
