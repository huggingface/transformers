"""Image helper for OCR using Pillow + pytesseract.

Note: pytesseract requires the Tesseract binary installed on the system.
On Windows, install Tesseract and add it to PATH, or set TESSERACT_CMD.
"""
from typing import Optional
from PIL import Image
import pytesseract


def image_to_text(path: str, lang: Optional[str] = None) -> str:
    """Extract text from image at `path`. Returns extracted text."""
    img = Image.open(path)
    config = ""
    try:
        if lang:
            text = pytesseract.image_to_string(img, lang=lang, config=config)
        else:
            text = pytesseract.image_to_string(img, config=config)
        return text.strip()
    finally:
        try:
            img.close()
        except Exception:
            pass
