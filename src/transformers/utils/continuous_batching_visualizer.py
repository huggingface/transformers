from threading import Event
from typing import Optional

from rich.text import Text
from rich.segment import Segment
from rich.style import Style
from textual.app import App, ComposeResult, RenderResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static, Footer, Header
from textual.strip import Strip
from textual.scroll_view import ScrollView
from textual.geometry import Size, Region
from textual.cache import LRUCache
from textual import work
from textual.message import Message
import torch




class AttentionMatrixWidget(ScrollView):
    """Widget to display attention matrix visualization."""

    def render(self) -> RenderResult:
        return "Attention Matrix will be displayed here."


class BatchContentsWidget(Widget):
    """Widget to display batch contents with caching."""

    tokens_to_display: reactive[Text] = reactive(Text("Batch contents will be displayed here."))

    def __init__(self):
        super().__init__()
        self._render_cache = LRUCache(maxsize=50)
        self._last_content_hash = None

    def render(self) -> RenderResult:
        # Cache rendered content based on content hash
        content_hash = hash(str(self.tokens_to_display))
        if content_hash == self._last_content_hash:
            # Defensive check to ensure cache exists and has get method
            if hasattr(self._render_cache, 'get'):
                try:
                    cached_result = self._render_cache.get(content_hash)
                    if cached_result is not None:
                        return cached_result
                except (AttributeError, TypeError):
                    # Cache might be in an inconsistent state, recreate it
                    self._render_cache = LRUCache(maxsize=50)
        
        self._last_content_hash = content_hash
        result = self.tokens_to_display
        
        # Defensive check before setting cache
        if hasattr(self._render_cache, 'set'):
            try:
                self._render_cache.set(content_hash, result)
            except (AttributeError, TypeError):
                # Cache might be in an inconsistent state, recreate it
                self._render_cache = LRUCache(maxsize=50)
                self._render_cache.set(content_hash, result)
        
        return result

class CacheWidget(Widget):
    """Widget to display cache information with static content optimization."""

    def render(self) -> RenderResult:
        """Render the cache information."""
        return "Cache usage will be shown here"


class ContinuousBatchingVisualizer(App):
    """Main application for visualizing continuous batching with performance optimizations."""

    # Bind 'q' key to quit action
    BINDINGS = [("n", "next", "Next"), ("q", "quit", "Quit")]

    CSS = """
    /* Top row widgets */
    #top-row {
        height: 65%;
    }
    
    AttentionMatrixWidget {
        width: 50%;
        border: solid #87CEEB;
        margin: 0;
        scrollbar-size: 1 1;
    }
    
    CacheWidget {
        width: 50%;
        border: solid #98FB98;
        margin: 0;
    }
    
    /* Bottom widget */
    BatchContentsWidget {
        width: 100%;
        height: 35%;
        border: solid #FFB6C1;
        margin: 0;
    }
    
    .content {
        padding: 1;
        background: $surface;
    }
    """

    def __init__(self):
        super().__init__()
        self.exited = False
        self.wait_event = Event()
        self._color_cache = LRUCache(maxsize=1024)
        self._data_processing_cache = LRUCache(maxsize=50)

    def compose(self) -> ComposeResult:
        """Compose the app layout."""
        yield Header()
        with Vertical():
            with Horizontal(id="top-row"):
                yield AttentionMatrixWidget()
                yield CacheWidget()
            yield BatchContentsWidget()
        yield Footer()

    def run(self):
        self.exited = False
        self.wait_event = Event()
        super().run()

    def action_quit(self) -> None:
        """Action to quit the application."""
        self.wait_event.set()
        self.exited = True
        self.exit()

    def action_next(self) -> None:
        """Action to update visualizations with new data."""
        self.wait_event.set()

    def draw(self, data):
        """Update all widgets with new data using background worker for performance."""
        if self.exited:
            return

        colored_text = Text()
        batch_contents = data.get("batch_contents", [])
        for i, token_info in enumerate(batch_contents):
            request_id = token_info.get("request_id", "unknown")
            color = self._get_cached_color(request_id)
            
            if "tokens" in token_info:
                text_content = " ".join(map(str, token_info["tokens"]))
            elif "decoded" in token_info:
                text_content = token_info["decoded"]
            else:
                text_content = ""
        
            if text_content:
                colored_text.append(f"[{request_id}] ", style=f"bold {color}")
                colored_text.append(text_content, style=color)
                
                if i < len(batch_contents) - 1:
                    colored_text.append(" | ", style="white")

        self.query_one(BatchContentsWidget).tokens_to_display = colored_text

    def _get_cached_color(self, request_id: str) -> str:
        """Get cached color for request ID."""
        cached_color = self._color_cache.get(request_id)
        if cached_color is not None:
            return cached_color

        r, g, b = self.string_to_rgb_color(request_id)
        cached_color = f"rgb({r},{g},{b})"

        return cached_color

    def wait_for_input(self):
        """Wait for user input to update visualizations."""
        # Implementation for waiting for user input
        if self.exited:
            return
        self.wait_event.wait()
        self.wait_event.clear()

    def string_to_rgb_color(self, input_string: str) -> tuple[int, int, int]:
        """Generate a consistent RGB color from an input string.
        
        Args:
            input_string: The string to convert to an RGB color
            
        Returns:
            A tuple of (r, g, b) values where each component is 0-255
        """
        # Use Python's built-in hash function for consistency
        hash_value = hash(input_string)
        
        # Convert to positive value and ensure we have enough bits
        hash_value = abs(hash_value)
        
        # Extract RGB components from different parts of the hash
        # Use bit shifting and masking to get 8-bit values for each channel
        r = (hash_value >> 16) & 0xFF
        g = (hash_value >> 8) & 0xFF  
        b = hash_value & 0xFF
        
        # Ensure colors are bright enough to be visible (minimum 64)
        # and not too bright (maximum 255)
        r = max(64, min(255, r))
        g = max(64, min(255, g))
        b = max(64, min(255, b))
        
        return (r, g, b)
