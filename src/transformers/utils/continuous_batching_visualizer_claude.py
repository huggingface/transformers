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
    """Widget to display attention matrix visualization using Line API for performance."""

    COMPONENT_CLASSES = {
        "attention--green-square",
        "attention--yellow-square", 
        "attention--black-square",
        "attention--white-square",
    }

    DEFAULT_CSS = """
    AttentionMatrixWidget .attention--green-square {
        color: green;
    }
    AttentionMatrixWidget .attention--yellow-square {
        color: yellow;
    }
    AttentionMatrixWidget .attention--black-square {
        color: white;
    }
    AttentionMatrixWidget .attention--white-square {
        color: #444444;
    }
    """

    def __init__(self):
        super().__init__()
        self.words = []
        self.mask = None
        self.img_token = "<img>"
        self.sliding_window = None
        self.token_type_ids = None
        self.image_seq_length = None
        self.max_word_length = 0
        self.header_lines = 0
        
        # Performance optimizations
        self._style_cache = LRUCache(maxsize=100)
        self._segment_cache = LRUCache(maxsize=1000)
        self._processed_mask = None
        self._last_data_hash = None
        
    def set_attention_data(self, words, mask, img_token="<img>", sliding_window=None, 
                          token_type_ids=None, image_seq_length=None):
        """Set the attention matrix data with caching optimization."""
        # Create a hash of the input data to check if anything changed
        data_hash = hash((
            tuple(words) if words else (),
            tuple(mask.flatten().tolist()) if mask is not None else (),
            img_token,
            sliding_window,
            tuple(token_type_ids) if token_type_ids is not None else None,
            image_seq_length
        ))
        
        # Skip processing if data hasn't changed
        if data_hash == self._last_data_hash:
            return
            
        self._last_data_hash = data_hash
        self._segment_cache.clear()  # Clear cache when data changes
        
        self.words = words
        self.mask = mask.int() if mask is not None else None
        self.img_token = img_token
        self.sliding_window = sliding_window
        self.token_type_ids = token_type_ids
        self.image_seq_length = image_seq_length
        
        if self.mask is not None:
            if self.mask.ndim == 3:
                self.mask = self.mask[0, :, :]
            if self.mask.ndim == 4:
                self.mask = self.mask[0, 0, :, :]
                
        if words:
            self.max_word_length = max(len(repr(word)) for word in words)
            self._process_mask()
            self._calculate_virtual_size()
        else:
            self.max_word_length = 0
            
        # Use region-based refresh for better performance
        if self.words:
            self.refresh_lines(0, len(self.words) + self.header_lines)
        else:
            self.refresh_lines(0, 10)
    
    def on_resize(self, event) -> None:
        """Handle resize events by clearing segment cache to prevent issues."""
        try:
            if hasattr(self, '_segment_cache') and hasattr(self._segment_cache, 'clear'):
                self._segment_cache.clear()
        except (AttributeError, TypeError):
            # Recreate cache if there are issues
            self._segment_cache = LRUCache(maxsize=1000)
        
    def _process_mask(self):
        """Process mask to mark image token regions with caching."""
        if self.mask is None or not self.words:
            self._processed_mask = None
            return
            
        # Create a copy to avoid modifying the original
        self._processed_mask = self.mask.clone()
        n = len(self.words)
        first_img_idx = 0
        
        for i, k in enumerate(self.words):
            if k == self.img_token and not first_img_idx:
                first_img_idx = i
                self._processed_mask[i, i] = 2  # Mark yellow regions
            if first_img_idx > 0 and (k != self.img_token or i == n - 1):
                if i == n - 1:
                    i += 1
                self._processed_mask[first_img_idx:i, first_img_idx:i] = 2  # Mark yellow regions
                first_img_idx = 0
                
    def _calculate_virtual_size(self):
        """Calculate the virtual size for scrolling."""
        if not self.words:
            self.virtual_size = Size(80, 10)
            return
            
        n = len(self.words)
        # Header lines: legend + title + vertical header
        self.header_lines = 2 + len(str(n))
        # Total height: headers + one line per word
        total_height = self.header_lines + n
        # Width: word column + attention matrix + optional sliding window
        matrix_width = n * 2 - 1  # spaces between squares
        total_width = self.max_word_length + 5 + matrix_width
        if self.sliding_window is not None:
            total_width += 8 + matrix_width  # separator + sliding window matrix
            
        self.virtual_size = Size(total_width, total_height)

    def _get_cached_styles(self):
        """Get cached component styles for better performance."""
        cache_key = "component_styles"
        try:
            if hasattr(self._style_cache, 'get'):
                styles = self._style_cache.get(cache_key)
                if styles is not None:
                    return styles
        except (AttributeError, TypeError):
            # Cache might be corrupted, recreate it
            self._style_cache = LRUCache(maxsize=100)
            
        styles = {
            'green': self.get_component_rich_style("attention--green-square"),
            'yellow': self.get_component_rich_style("attention--yellow-square"),
            'black': self.get_component_rich_style("attention--black-square"),
            'white': self.get_component_rich_style("attention--white-square")
        }
        
        try:
            if hasattr(self._style_cache, 'set'):
                self._style_cache.set(cache_key, styles)
        except (AttributeError, TypeError):
            # Cache might be corrupted, recreate it
            self._style_cache = LRUCache(maxsize=100)
            self._style_cache.set(cache_key, styles)
            
        return styles

    def render_line(self, y: int) -> Strip:
        """Render a single line of the attention matrix with optimizations."""
        if not self.words or self._processed_mask is None:
            return Strip([Segment("Attention mask will be displayed here.")])
        
        # Check segment cache first with defensive programming
        cache_key = (y, self.scroll_offset.x, self.scroll_offset.y, self.size.width)
        try:
            if hasattr(self._segment_cache, 'get'):
                cached_strip = self._segment_cache.get(cache_key)
                if cached_strip is not None:
                    return cached_strip
        except (AttributeError, TypeError):
            # Cache might be corrupted, recreate it
            self._segment_cache = LRUCache(maxsize=1000)
            
        scroll_x, scroll_y = self.scroll_offset
        y += scroll_y
        n = len(self.words)
        
        # Get cached styles
        styles = self._get_cached_styles()
        
        # Define symbols as constants
        BLACK_SQUARE = "■"
        WHITE_SQUARE = "⬚"
        
        if y < self.header_lines:
            strip = self._render_header_line(y, n, styles['green'], styles['yellow'], BLACK_SQUARE)
        elif y - self.header_lines < n:
            row_idx = y - self.header_lines
            strip = self._render_matrix_row(row_idx, n, styles, BLACK_SQUARE, WHITE_SQUARE)
        else:
            strip = Strip.blank(self.size.width)
        
        # Cache the result with defensive programming
        try:
            if hasattr(self._segment_cache, 'set'):
                self._segment_cache.set(cache_key, strip)
        except (AttributeError, TypeError):
            # Cache might be corrupted, recreate it
            self._segment_cache = LRUCache(maxsize=1000)
            self._segment_cache.set(cache_key, strip)
        
        return strip
    
    def _render_header_line(self, y: int, n: int, green_style: Style, yellow_style: Style, BLACK_SQUARE: str) -> Strip:
        """Render header lines (legend, title, vertical indices) with cropping optimization."""
        scroll_x, _ = self.scroll_offset
        
        if y == 0:
            # Legend line
            segments = [
                Segment(" ", style=Style()),
                Segment(BLACK_SQUARE, style=green_style),
                Segment(": i == j (diagonal)   ", style=Style()),
                Segment(BLACK_SQUARE, style=yellow_style),
                Segment(": token_type_ids", style=Style())
            ]
        elif y == 1:
            # Title line
            padding = " " * (self.max_word_length + 5)
            title = "Attention Matrix"
            if self.sliding_window is not None:
                title += "".ljust(max(0, (n * 2 - 1) // 2 - len(title) // 2)) + "Sliding Window Mask"
            segments = [Segment(padding + title, style=Style())]
        else:
            # Vertical header lines (column indices)
            header_row = y - 2
            if header_row < len(str(n)):
                segments = [Segment(" " * (self.max_word_length + 5), style=Style())]
                
                # Generate column indices for this row
                for col_idx in range(n):
                    idx_str = str(col_idx).rjust(len(str(n)))
                    if header_row < len(idx_str):
                        char = idx_str[header_row]
                        style = yellow_style if self._processed_mask[col_idx, col_idx] == 2 else Style()
                        segments.append(Segment(char, style=style))
                    else:
                        segments.append(Segment(" ", style=Style()))
                    
                    if col_idx < n - 1:
                        segments.append(Segment(" ", style=Style()))
                
                if self.sliding_window is not None:
                    segments.append(Segment("    |    ", style=Style()))
                    # Repeat for sliding window side
                    for col_idx in range(n):
                        idx_str = str(col_idx).rjust(len(str(n)))
                        if header_row < len(idx_str):
                            char = idx_str[header_row]
                            segments.append(Segment(char, style=Style()))
                        else:
                            segments.append(Segment(" ", style=Style()))
                        
                        if col_idx < n - 1:
                            segments.append(Segment(" ", style=Style()))
            else:
                return Strip.blank(self.size.width)
        
        strip = Strip(segments)
        return strip.crop(scroll_x, scroll_x + self.size.width)
    
    def _render_matrix_row(self, row_idx: int, n: int, styles: dict, BLACK_SQUARE: str, WHITE_SQUARE: str) -> Strip:
        """Render a single row of the attention matrix with optimizations."""
        scroll_x, _ = self.scroll_offset
        
        word = self.words[row_idx]
        word_repr = repr(word).ljust(self.max_word_length)
        
        segments = []
        
        # Word label
        word_style = styles['yellow'] if self.img_token in word else Style()
        segments.append(Segment(word_repr, style=word_style))
        segments.append(Segment(f": {str(row_idx).rjust(2)} ", style=Style()))
        
        # Attention matrix row - batch process for better performance
        for col_idx in range(n):
            mask_val = self._processed_mask[row_idx, col_idx]
            col_word = self.words[col_idx]
            
            if (self.img_token in col_word and mask_val and self.img_token in word):
                segments.append(Segment(BLACK_SQUARE, style=styles['yellow']))
            elif row_idx == col_idx:
                segments.append(Segment(BLACK_SQUARE, style=styles['green']))
            elif mask_val:
                segments.append(Segment(BLACK_SQUARE, style=styles['black']))
            else:
                segments.append(Segment(WHITE_SQUARE, style=styles['white']))
                
            if col_idx < n - 1:
                segments.append(Segment(" ", style=Style()))
        
        # Sliding window matrix (if enabled)
        if self.sliding_window is not None:
            segments.append(Segment("    |    ", style=Style()))
            
            # Pre-compute sliding window mask for the row
            sliding_start = max(0, row_idx - self.sliding_window + 1)
            sliding_end = min(n, row_idx + 1)
            
            for col_idx in range(n):
                col_word = self.words[col_idx]
                in_sliding_window = sliding_start <= col_idx < sliding_end
                
                if (self.token_type_ids is not None and
                    self.img_token in col_word and self.img_token in word):
                    segments.append(Segment(BLACK_SQUARE, style=styles['yellow']))
                elif row_idx == col_idx:
                    segments.append(Segment(BLACK_SQUARE, style=styles['green']))
                elif in_sliding_window:
                    segments.append(Segment(BLACK_SQUARE, style=styles['black']))
                else:
                    segments.append(Segment(WHITE_SQUARE, style=styles['white']))
                    
                if col_idx < n - 1:
                    segments.append(Segment(" ", style=Style()))
        
        strip = Strip(segments)
        return strip.crop(scroll_x, scroll_x + self.size.width)


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
    
    def on_resize(self, event) -> None:
        """Handle resize events by clearing render cache to prevent issues."""
        try:
            if hasattr(self, '_render_cache') and hasattr(self._render_cache, 'clear'):
                self._render_cache.clear()
            self._last_content_hash = None
        except (AttributeError, TypeError):
            # Recreate cache if there are issues
            self._render_cache = LRUCache(maxsize=50)
            self._last_content_hash = None

class CacheWidget(Static):
    """Widget to display cache information with static content optimization."""
    
    def __init__(self):
        super().__init__("Cache visualization content here", classes="content")
        self._cache_info = {}
        self._display_cache = LRUCache(maxsize=20)
    
    def update_cache_info(self, cache_info: dict):
        """Update cache information efficiently."""
        if cache_info != self._cache_info:
            self._cache_info = cache_info.copy()
            cache_text = self._format_cache_info(cache_info)
            self.update(cache_text)
    
    def _format_cache_info(self, cache_info: dict) -> str:
        """Format cache information for display."""
        cache_key = hash(str(cache_info))
        try:
            if hasattr(self._display_cache, 'get'):
                cached_text = self._display_cache.get(cache_key)
                if cached_text is not None:
                    return cached_text
        except (AttributeError, TypeError):
            # Cache might be corrupted, recreate it
            self._display_cache = LRUCache(maxsize=20)
            
        lines = ["Cache Information:"]
        for key, value in cache_info.items():
            lines.append(f"  {key}: {value}")
        
        result = "\n".join(lines)
        
        try:
            if hasattr(self._display_cache, 'set'):
                self._display_cache.set(cache_key, result)
        except (AttributeError, TypeError):
            # Cache might be corrupted, recreate it
            self._display_cache = LRUCache(maxsize=20)
            self._display_cache.set(cache_key, result)
            
        return result


class ContinuousBatchingVisualizer(App):
    """Main application for visualizing continuous batching with performance optimizations."""
    
    class UpdateWidgets(Message):
        """Message to update widgets with processed data."""
        def __init__(self, processed_data: dict):
            super().__init__()
            self.processed_data = processed_data
    
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

    def on_update_widgets(self, message: UpdateWidgets) -> None:
        """Handle UpdateWidgets message by updating widgets with processed data."""
        self._update_widgets(message.processed_data)

    @work(exclusive=True)
    async def draw(self, data):
        """Update all widgets with new data using background worker for performance."""
        if self.exited:
            return
        
        # Process data in background to avoid UI blocking
        processed_data = await self._process_data_async(data)
        
        # Update UI from worker thread using post_message
        self.post_message(self.UpdateWidgets(processed_data))
    
    async def _process_data_async(self, data):
        """Process data asynchronously to avoid blocking UI."""
        # Create a hash to check if we can use cached processing
        data_hash = hash(str(data))
        try:
            if hasattr(self._data_processing_cache, 'get'):
                cached_result = self._data_processing_cache.get(data_hash)
                if cached_result is not None:
                    return cached_result
        except (AttributeError, TypeError):
            # Cache might be corrupted, recreate it
            self._data_processing_cache = LRUCache(maxsize=50)
        
        # Create a Rich Text object to hold colored tokens
        colored_text = Text()
        attn_mask_words = []
        
        batch_contents = data.get("batch_contents", [])
        for i, token_info in enumerate(batch_contents):
            request_id = token_info.get("request_id", "unknown")
            
            # Get cached RGB color for this request ID
            color = self._get_cached_color(request_id)
            
            # Extract tokens or decoded text
            if "tokens" in token_info:
                text_content = " ".join(map(str, token_info["tokens"]))
            elif "decoded" in token_info:
                text_content = token_info["decoded"]
                attn_mask_words.extend(token_info["decoded_tokens"])
            else:
                text_content = ""
            
            # Add colored text with request ID prefix
            if text_content:
                colored_text.append(f"[{request_id}] ", style=f"bold {color}")
                colored_text.append(text_content, style=color)
                
                # Add separator if not the last item
                if i < len(batch_contents) - 1:
                    colored_text.append(" | ", style="white")

        result = {
            'colored_text': colored_text,
            'attn_mask_words': attn_mask_words,
            'attention_mask': data.get("attention_mask", torch.zeros((len(attn_mask_words), len(attn_mask_words)))) if attn_mask_words else None,
            'sliding_window': data.get("sliding_window"),
            'token_type_ids': data.get("token_type_ids"),
            'image_seq_length': data.get("image_seq_length")
        }
        
        # Cache the processed result with defensive programming
        try:
            if hasattr(self._data_processing_cache, 'set'):
                self._data_processing_cache.set(data_hash, result)
        except (AttributeError, TypeError):
            # Cache might be corrupted, recreate it
            self._data_processing_cache = LRUCache(maxsize=50)
            self._data_processing_cache.set(data_hash, result)
            
        return result
    
    def _update_widgets(self, processed_data):
        """Update widgets with processed data."""
        # Update attention matrix widget with new data
        if processed_data['attn_mask_words']:
            self.query_one(AttentionMatrixWidget).set_attention_data(
                processed_data['attn_mask_words'],
                processed_data['attention_mask'],
                img_token="<img>",
                sliding_window=processed_data['sliding_window'],
                token_type_ids=processed_data['token_type_ids'],
                image_seq_length=processed_data['image_seq_length']
            )

        # Update the batch contents widget
        self.query_one(BatchContentsWidget).tokens_to_display = processed_data['colored_text']
        
        # Update cache widget with performance metrics
        cache_info = {
            "Color Cache Size": len(self._color_cache.keys()),
            "Data Cache Size": len(self._data_processing_cache.keys()),
            "Matrix Cache": len(self.query_one(AttentionMatrixWidget)._segment_cache.keys()),
        }
        self.query_one(CacheWidget).update_cache_info(cache_info)
    
    def _get_cached_color(self, request_id: str) -> str:
        """Get cached color for request ID."""
        try:
            if hasattr(self._color_cache, 'get'):
                cached_color = self._color_cache.get(request_id)
                if cached_color is not None:
                    return cached_color
        except (AttributeError, TypeError):
            # Cache might be corrupted, recreate it
            self._color_cache = LRUCache(maxsize=1024)
            
        r, g, b = self.string_to_rgb_color(request_id)
        cached_color = f"rgb({r},{g},{b})"
        
        try:
            if hasattr(self._color_cache, 'set'):
                self._color_cache.set(request_id, cached_color)
        except (AttributeError, TypeError):
            # Cache might be corrupted, recreate it
            self._color_cache = LRUCache(maxsize=1024)
            self._color_cache.set(request_id, cached_color)
            
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
