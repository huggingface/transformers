from threading import Event
from typing import Optional, List, Any, Dict
import hashlib

from rich.text import Text
from rich.segment import Segment
from rich.style import Style
from textual.app import App, ComposeResult, RenderResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static, Footer, Header, RichLog
from textual.strip import Strip
from textual.scroll_view import ScrollView
from textual.geometry import Size
from textual.cache import LRUCache
import torch

# Constants for visualization
BLACK_SQUARE = "■"
WHITE_SQUARE = "⬚"


class AttentionMatrixWidget(ScrollView):
    """Widget to display attention matrix visualization with request ID-based coloring."""
    
    DEFAULT_CSS = """
    AttentionMatrixWidget {
        scrollbar-size: 1 1;
    }
    """

    def __init__(self):
        super().__init__()
        
        # Attention matrix data
        self.words: List[str] = []
        self.mask: Optional[torch.Tensor] = None
        self.request_ids: List[str] = []  # Request ID for each token
        self.img_token: str = "<img>"
        
        # Processed data for rendering
        self._processed_mask: Optional[torch.Tensor] = None
        self._max_word_length: int = 0
        self.header_lines: int = 0
        
        # Performance caches
        self._segment_cache = LRUCache(maxsize=1000)
        self._style_cache = LRUCache(maxsize=100)
        self._data_hash: Optional[str] = None
        
        # Color scheme for request IDs
        self._color_cache = LRUCache(maxsize=100)
        
    def set_attention_data(
        self,
        words: List[str],
        mask: torch.Tensor,
        request_ids: Optional[List[str]] = None,
        img_token: str = "<img>",
        **kwargs
    ):
        """Set new attention data and trigger re-rendering."""
        # Create hash of input data for caching
        data_str = f"{words}_{mask.shape}_{request_ids}_{img_token}"
        new_hash = hashlib.md5(data_str.encode()).hexdigest()
        
        # Always update if data has changed or if this is first time
        if new_hash != self._data_hash or self._data_hash is None:
            self._data_hash = new_hash
            
            # Clear caches when data changes
            self._segment_cache.clear()
            self._style_cache.clear()
            
            # Store raw data
            self.words = words
            self.mask = mask.clone()
            self.request_ids = request_ids or ["unknown"] * len(words)
            self.img_token = img_token
            
            # Process the data
            self._process_attention_data()
            
            # Update virtual size and refresh
            self._calculate_virtual_size()
            self.refresh()
        
    def _process_attention_data(self):
        """Process attention data for efficient rendering."""
        if not self.words or self.mask is None:
            return
            
        # Convert mask to 2D
        mask = self.mask.int()
        
        if mask.ndim == 3:
            mask = mask[0, :, :]
        elif mask.ndim == 4:
            mask = mask[0, 0, :, :]
            
        n = len(self.words)
        self._max_word_length = max(len(repr(word)) for word in self.words) if self.words else 0
        
        self._processed_mask = mask

    def _calculate_virtual_size(self):
        """Calculate the virtual size for scrolling."""
        if not self.words:
            virtual_height = 1
        else:
            virtual_height = len(self.words)
            
        # Width based on content (word length + matrix + spacing)
        if self.words:
            matrix_width = len(self.words) * 2  # Each cell takes 2 chars (symbol + space)
            virtual_width = self._max_word_length + 10 + matrix_width
        else:
            virtual_width = 50
            
        self.virtual_size = Size(virtual_width, virtual_height)
        
    def _get_request_id_color(self, request_id: str) -> Style:
        """Get cached color style for request ID."""
        cached_style = self._color_cache.get(request_id)
        if cached_style is not None:
            return cached_style
            
        # Generate consistent color for request ID
        r, g, b = self._string_to_rgb_color(request_id)
        color_str = f"rgb({r},{g},{b})"
        style = Style(color=color_str)
        
        self._color_cache.set(request_id, style)
        return style
        
    def _string_to_rgb_color(self, input_string: str) -> tuple[int, int, int]:
        """Generate a consistent RGB color from an input string."""
        hash_value = abs(hash(input_string))
        
        # Extract RGB components
        r = (hash_value >> 16) & 0xFF
        g = (hash_value >> 8) & 0xFF  
        b = hash_value & 0xFF
        
        # Ensure colors are bright enough to be visible
        r = max(64, min(255, r))
        g = max(64, min(255, g))
        b = max(64, min(255, b))
        
        return (r, g, b)
    
    def render_line(self, y: int) -> Strip:
        """Render a single line using Line API for performance."""
        # Early return for empty data
        if not self.words or self._processed_mask is None:
            return Strip([Segment("No attention data to display", Style(color="gray50"))])
        
        # Get the actual content line based on viewport position
        content_line = y
        
        # Use a lighter caching approach - cache by content line and data hash only
        # Don't cache if we don't have stable data to avoid scroll interference
        cache_key = f"line_{content_line}_{self._data_hash}" if self._data_hash else None
        cached_strip = None
        if cache_key:
            cached_strip = self._segment_cache.get(cache_key)
        if cached_strip is not None:
            return cached_strip
            
        n = len(self.words)
        
        # Render different types of lines based on content position
        if content_line == 0:
            strip = self._render_title_line()
        elif content_line < n:
            # Matrix row
            strip = self._render_matrix_row(content_line)
        else:
            # Empty line
            strip = Strip([Segment("")])
            
        # Cache the result only if we have a valid cache key
        if cache_key:
            self._segment_cache.set(cache_key, strip)
        return strip

    def _render_title_line(self) -> Strip:
        """Render the title line."""
        title = f"Attention Matrix ({len(self.words)}x{len(self.words)})"
        return Strip([Segment(title, Style(bold=True))])

    def _render_matrix_row(self, row_idx: int) -> Strip:
        """Render a single matrix row with request ID-based coloring."""
        if row_idx >= len(self.words) or self._processed_mask is None:
            return Strip([Segment("")])
            
        word = self.words[row_idx]
        word_repr = repr(word).ljust(self._max_word_length)
        
        segments = []
        
        # Row label (word) - colored by request ID
        row_request_id = self.request_ids[row_idx] if row_idx < len(self.request_ids) else "unknown"
        row_style = self._get_request_id_color(row_request_id)
        segments.append(Segment(word_repr, row_style))
        segments.append(Segment(f": {str(row_idx).rjust(2)} ", Style()))
        
        # Matrix cells
        for col_idx in range(len(self.words)):
            mask_value = self._processed_mask[row_idx, col_idx].item()
            col_request_id = self.request_ids[col_idx] if col_idx < len(self.request_ids) else "unknown"
            
            if mask_value == 1:  # Attended - use request ID color
                symbol = BLACK_SQUARE
                # Use the color of the target request ID (column)
                style = self._get_request_id_color(col_request_id)
            else:  # Not attended
                symbol = WHITE_SQUARE
                style = Style(color="gray50")
                
            segments.append(Segment(symbol, style))
            segments.append(Segment(" ", Style()))  # Spacing
            
        return Strip(segments)




class BatchContentsWidget(RichLog):
    """Widget to display batch contents with request ID coloring using RichLog."""

    DEFAULT_CSS = """
    BatchContentsWidget {
        height: 35%;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(
            auto_scroll=False,
            markup=True,
            wrap=True,
            **kwargs
        )

    def set_batch_contents(self, batch_contents: List[Dict[str, Any]]):
        """Set batch contents and update display."""
        # Clear existing content
        self.clear()
        
        if not batch_contents:
            self.write("Batch contents will be displayed here.")
            return
            
        # Write each token info as a separate line
        for token_info in batch_contents:
            request_id = token_info.get("request_id", "unknown")
            color = self._get_color_for_request(request_id)
            
            # Create Rich Text for this token
            token_text = Text()
            token_text.append(f"[{request_id}] ", style=f"bold {color}")
            
            if "decoded" in token_info:
                token_text.append(token_info["decoded"], style=color)
            elif "tokens" in token_info:
                tokens_str = " ".join(map(str, token_info["tokens"]))
                token_text.append(tokens_str, style=color)
            else:
                token_text.append("(no content)", style=color)
            
            # Write the token info to the log
            self.write(token_text)

    def _get_color_for_request(self, request_id: str) -> str:
        """Get color for request ID - delegates to parent app."""
        app = self.app
        if hasattr(app, '_get_cached_color'):
            return app._get_cached_color(request_id)
        return "white"  # fallback


class CacheWidget(Widget):
    """Widget to display PagedAttentionCache contents and statistics."""

    cache_info: reactive[Text] = reactive(Text("PagedAttentionCache: waiting for data..."))

    def render(self) -> RenderResult:
        return self.cache_info


class ContinuousBatchingVisualizer(App):
    """Main application for visualizing continuous batching with request ID-based coloring."""

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
        self._pending_attention_data = None

    def compose(self) -> ComposeResult:
        """Compose the app layout."""
        yield Header()
        with Vertical():
            with Horizontal(id="top-row"):
                yield AttentionMatrixWidget()
                yield CacheWidget()
            yield BatchContentsWidget()
        yield Footer()

    def on_mount(self) -> None:
        """Called when the app is mounted and widgets are available."""
        # If we have pending attention data, apply it now
        if self._pending_attention_data:
            self.set_timer(0.1, self._apply_pending_attention_data)

    def _apply_pending_attention_data(self) -> None:
        """Apply any pending attention data if widgets are ready."""
        if self._pending_attention_data:
            try:
                attention_widget = self.query_one(AttentionMatrixWidget)
                attention_widget.set_attention_data(**self._pending_attention_data)
                self._pending_attention_data = None
            except Exception:
                # Try again later if widget still not ready
                self.set_timer(0.1, self._apply_pending_attention_data)

    def action_quit(self) -> None:
        """Action to quit the application."""
        self.wait_event.set()
        self.exited = True
        self.exit()

    def action_next(self) -> None:
        """Action to update visualizations with new data."""
        self.wait_event.set()

    def draw(self, data: Dict[str, Any]):
        """
        Update all widgets with new data from continuous batching.
        
        Expected data format:
        {
            'batch_contents': [
                {
                    'request_id': str,
                    'tokens': List[int] or 'decoded': str,
                    'decoded_tokens': List[str]  # optional
                }
            ],
            'attention_mask': torch.Tensor,
            'words': List[str],  # tokens as strings
            'request_ids_per_token': List[str]  # request ID for each token
        }
        """
        if self.exited:
            return

        try:
            # Update batch contents widget
            self._update_batch_contents(data.get('batch_contents', []))
            
            # Update attention matrix widget
            self._update_attention_matrix(data)
            
            # Update cache info
            self._update_cache_info(data)
            
        except Exception as e:
            # Display error in cache widget
            cache_widget = self.query_one(CacheWidget)
            cache_widget.cache_info = Text(f"Error: {str(e)}", style="red")

    def _update_batch_contents(self, batch_contents: List[Dict[str, Any]]):
        """Update the batch contents widget with scrollable display."""
        try:
            batch_widget = self.query_one(BatchContentsWidget)
            batch_widget.set_batch_contents(batch_contents)
        except Exception:
            pass  # Widget not ready yet

    def _update_attention_matrix(self, data: Dict[str, Any]):
        """Update the attention matrix widget."""
        words = data.get('words', [])
        attention_mask = data.get('attention_mask')
        request_ids = data.get('request_ids_per_token', [])
        
        if words and attention_mask is not None:
            try:
                attention_widget = self.query_one(AttentionMatrixWidget)
                attention_widget.set_attention_data(
                    words=words,
                    mask=attention_mask,
                    request_ids=request_ids
                )
            except Exception as e:
                # If we can't find the widget, store the data and try again later
                self._pending_attention_data = {
                    'words': words,
                    'mask': attention_mask,
                    'request_ids': request_ids
                }
                # Try again in a bit
                self.set_timer(0.1, self._apply_pending_attention_data)

    def _update_cache_info(self, data: Dict[str, Any]):
        """Update cache information display."""
        cache_data = data.get('paged_attention_cache', {})
        
        # Format PagedAttentionCache stats
        cache_lines = ["[bold green]PagedAttentionCache[/bold green]"]
        if cache_data:
            # Display key PagedAttentionCache metrics
            cache_lines.extend([
                f"Total blocks: {cache_data.get('total_blocks', 0)}",
                f"Used blocks: {cache_data.get('used_blocks', 0)}",
                f"Free blocks: {cache_data.get('free_blocks', 0)}",
                f"Block size: {cache_data.get('block_size', 'Unknown')}",
                f"Num heads: {cache_data.get('num_heads', 'Unknown')}",
                f"Head dim: {cache_data.get('head_dim', 'Unknown')}",
            ])
            
            # Show utilization if available
            if 'utilization' in cache_data:
                cache_lines.append(f"Utilization: {cache_data['utilization']:.1%}")
        else:
            cache_lines.append("No PagedAttentionCache data available")

        cache_info = Text.from_markup("\n".join(cache_lines))

        try:
            cache_widget = self.query_one(CacheWidget)
            cache_widget.cache_info = cache_info
            
        except Exception:
            # Widget not ready yet, just show basic info
            try:
                cache_widget = self.query_one(CacheWidget)
                cache_info = Text("Cache info loading...", style="yellow")
                cache_widget.cache_info = cache_info
            except Exception:
                pass  # CacheWidget not ready either

    def _get_cached_color(self, request_id: str) -> str:
        """Get cached color for request ID (same as attention matrix)."""
        cached_color = self._color_cache.get(request_id)
        if cached_color is not None:
            return cached_color

        r, g, b = self._string_to_rgb_color(request_id)
        cached_color = f"rgb({r},{g},{b})"
        self._color_cache.set(request_id, cached_color)
        return cached_color

    def _string_to_rgb_color(self, input_string: str) -> tuple[int, int, int]:
        """Generate a consistent RGB color from an input string."""
        hash_value = abs(hash(input_string))
        
        # Extract RGB components
        r = (hash_value >> 16) & 0xFF
        g = (hash_value >> 8) & 0xFF  
        b = hash_value & 0xFF
        
        # Ensure colors are bright enough to be visible
        r = max(64, min(255, r))
        g = max(64, min(255, g))
        b = max(64, min(255, b))
        
        return (r, g, b)

    def wait_for_input(self):
        """Wait for user input to update visualizations."""
        if self.exited:
            return
        self.wait_event.wait()
        self.wait_event.clear()
