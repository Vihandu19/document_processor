import pdfplumber
import regex
from io import BytesIO
import logging

from typing import List, Dict, Any
from itertools import groupby

logger = logging.getLogger(__name__)

# Extract text from PDF file
def parse_pdf(file_contents: bytes) -> str:
    try:
        
        pdf_file = BytesIO(file_contents)
        
        
        text_parts = []
        
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text and text.strip():  # Only add non-empty pages
                    text_parts.append(text)
                    logger.debug(f"Extracted {len(text)} chars from page {page_num}")
        
        full_text = "\n\n".join(text_parts)
        logger.info(f"Successfully extracted {len(full_text)} characters from PDF")
        
        return full_text
    
    except Exception as e:
        logger.error(f"Failed to parse PDF: {str(e)}")
        raise ValueError(f"Could not parse PDF: {str(e)}")

# Extract detailed line features from PDF
def extract_and_parse_pdf(file_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extract text-line features from a PDF using pdfplumber.
    Each line is assigned a stable visual signature (line_signature)
    used for detecting repeated headers/footers across pages.
    """
    pdf_file = BytesIO(file_bytes)
    all_lines_data = []

    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages):

                # Extract words with font metadata
                words = page.extract_words(
                    x_tolerance=3,
                    y_tolerance=3,
                    extra_attrs=["fontname", "size"]
                )

                if not words:
                    continue

                # Line signature: groups visually identical lines across pages
                line_key = lambda w: (
                    round(w["top"], 1),                 # vertical location
                    w.get("fontname", "UNKNOWN"),       # font face
                    round(w.get("size", 0), 1)          # font size
                )

                # sort before groupby (groupby only groups consecutive items)
                words = sorted(words, key=line_key)

                # Group words into lines
                for signature, group in groupby(words, key=line_key):
                    line_words = list(group)

                    # Compute line-level layout geometry
                    x0 = min(w["x0"] for w in line_words)
                    x1 = max(w["x1"] for w in line_words)
                    top = min(w["top"] for w in line_words)

                    # Reconstruct text
                    text = " ".join(w["text"] for w in line_words)

                    # Simple page-number pattern check
                    is_numeric_pattern = bool(
                        regex.match(r'^\s*[\(\[]?\s*\d+\s*[\)\]]?\s*$', text.strip())
                    )

                    # Aggregate extracted features
                    features = {
                        "page_num": page_num + 1,
                        "line_signature": str(signature),  #grouping key

                        "text_content": text,
                        "char_count": len(text),

                        # Layout
                        "x0": x0,
                        "x1": x1,
                        "y_pos_abs": top,
                        "page_height": page.height,
                        "y_pos_norm": top / page.height,

                        # Style
                        "fontname": signature[1],
                        "font_size": signature[2],

                        # Heuristics
                        "is_numeric_pattern": is_numeric_pattern,
                        "is_short": len(text) < 30,
                    }

                    all_lines_data.append(features)

        return all_lines_data

    except Exception as e:
        print(f"PDF extraction error: {e}")
        return []
