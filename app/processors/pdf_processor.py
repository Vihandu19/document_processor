import pdfplumber
import regex
import time
import logging
import unicodedata

from typing import List, Dict, Any
from io import BytesIO

logger = logging.getLogger(__name__)


def reconstruct_line_from_words(
    line: Dict[str, Any],
    words: List[Dict[str, Any]],
    y_tol: float = 2.0,
) -> str:
    """
    Reconstruct a line's text by joining pdfplumber word objects that vertically
    overlap the given line. This reliably restores spaces for LaTeX PDFs.
    """
    top = line["top"]
    bottom = line["bottom"]

    line_words = [
        w for w in words
        if w["top"] >= top - y_tol and w["bottom"] <= bottom + y_tol
    ]

    # Sort left-to-right just in case
    line_words.sort(key=lambda w: w["x0"])

    return " ".join(w["text"] for w in line_words).strip()


def pre_process_text(text: str) -> str:
    """
    Pre-processing normalization.
    Safe Unicode cleanup only (no spacing or semantic changes).
    """
    if not text:
        return text

    # Canonical Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Normalize non-breaking spaces
    text = text.replace("\u00A0", " ")

    # Remove zero-width / invisible characters
    text = regex.sub(r"[\u200B-\u200D\uFEFF]", "", text)

    return text


def extract_and_parse_pdf_linebyline(
    file_bytes: bytes,
    x_tolerance: float = 8,
    y_tolerance: float = 5,
) -> List[Dict[str, Any]]:
    """
    Extract text-line features from a PDF using a hybrid approach:

    - extract_text_lines(): layout, geometry, font, repetition
    - extract_words(): reliable word boundaries and spacing

    This handles LaTeX-generated PDFs correctly while preserving layout features
    for ML-based structural classification.
    """
    start_time = time.perf_counter()
    pdf_file = BytesIO(file_bytes)
    all_lines_data: List[Dict[str, Any]] = []

    try:
        with pdfplumber.open(pdf_file) as pdf:
            total_pages = len(pdf.pages)

            for page_num, page in enumerate(pdf.pages):
                # Layout-based line detection
                lines = page.extract_text_lines(
                    layout=True,
                    use_text_flow=True,
                    x_tolerance=x_tolerance,
                    y_tolerance=y_tolerance,
                )

                # Word extraction for correct spacing
                words = page.extract_words(
                    use_text_flow=True,
                    keep_blank_chars=False,
                    x_tolerance=1,
                    y_tolerance=3,
                )

                for line in lines:
                    # Prefer word-based reconstruction
                    text = reconstruct_line_from_words(line, words)

                    if not text:
                        text = line.get("text", "").strip()

                    #pre processing normalization
                    text = pre_process_text(text)

                    # Skip empty or noise lines
                    if not text or (len(text) == 1 and text.isalpha()):
                        continue

                    x0 = line["x0"]
                    x1 = line["x1"]
                    top = line["top"]

                    chars = line.get("chars", [])
                    if chars:
                        fontname = chars[0].get("fontname", "UNKNOWN")
                        font_size = chars[0].get("size", 10.0)
                    else:
                        fontname = "UNKNOWN"
                        font_size = 10.0

                    features = {
                        # Identifiers
                        "page_num": page_num + 1,
                        "line_signature": f"{round(top, 0)}_{fontname}_{round(font_size, 1)}",
                        "text_content": text,

                        # Layout features
                        "x0": x0,
                        "x1": x1,
                        "y_pos_abs": top,
                        "page_height": page.height,
                        "page_width": page.width,
                        "y_pos_norm": top / page.height,
                        "x_pos_norm": x0 / page.width,
                        "line_width": x1 - x0,
                        "line_width_norm": (x1 - x0) / page.width,
                        "is_centered": abs((x0 + x1) / 2 - page.width / 2) < (page.width * 0.1),

                        # Text features
                        "char_count": len(text),
                        "word_count": len(text.split()),
                        "is_short": len(text) < 30,
                        "is_very_short": len(text) < 10,
                        "is_single_word": len(text.split()) == 1,

                        # Typographic features
                        "fontname": fontname,
                        "font_size": font_size,
                        "is_bold": "bold" in fontname.lower(),
                        "is_italic": "italic" in fontname.lower(),

                        # Content features
                        "is_all_caps": text.isupper() and len(text) > 2,
                        "is_title_case": text.istitle(),
                        "is_numeric_pattern": bool(regex.match(r"^\s*[\(\[]?\s*\d+\s*[\)\]]?\s*$", text)),
                        "has_page_keyword": bool(regex.search(r"\bpage\b", text, regex.IGNORECASE)),
                        "has_date_pattern": bool(regex.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", text)),
                        "starts_with_number": bool(regex.match(r"^\d+[\.)]\s", text)),
                        "ends_with_colon": text.endswith(":"),

                        # Positional features
                        "is_top_10_percent": (top / page.height) < 0.1,
                        "is_bottom_10_percent": (top / page.height) > 0.9,
                        "is_first_page": page_num == 0,
                        "is_last_page": page_num == total_pages - 1,
                    }

                    all_lines_data.append(features)

            # Repetition / header-footer features
            signature_counts: Dict[str, int] = {}
            for line in all_lines_data:
                sig = line["line_signature"]
                signature_counts[sig] = signature_counts.get(sig, 0) + 1

            for line in all_lines_data:
                sig = line["line_signature"]
                line["signature_frequency"] = signature_counts[sig]
                line["appears_on_multiple_pages"] = signature_counts[sig] > 1
                line["appears_on_most_pages"] = signature_counts[sig] >= (total_pages * 0.6)

        logger.debug(f"PDF extracted in {time.perf_counter() - start_time:.2f}s")
        return all_lines_data

    except Exception as e:
        logger.exception("PDF extraction failed")
        return []


def extract_and_parse_pdf(pdf_path: str) -> List[Dict]:
    all_lines_data = []
    start_time = time.perf_counter()
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, 1):
                # USE THIS INSTEAD: It finds lines automatically based on visual alignment
                lines = page.extract_text_lines(layout=True, y_tolerance=3)
                
                for line in lines:
                    text = line["text"].strip()
                    if not text: continue

                    # Basic geometry from the line object
                    x0, top, x1, bottom = line["x0"], line["top"], line["x1"], line["bottom"]
                    
                    # Calculate features exactly like your model expects
                    line_data = {
                        "page_num": page_num,
                        "text_content": text,
                        "x0": x0,
                        "x1": x1,
                        "y_pos_abs": top,
                        "y_pos_norm": top / float(page.height),
                        "font_size": line.get("chars", [{}])[0].get("size", 10),
                        "fontname": line.get("chars", [{}])[0].get("fontname", "Unknown"),
                        # ... include all other boolean features your model needs here ...
                    }
                    
                    # Add your custom logic for bold/caps/etc
                    line_data["is_bold"] = "Bold" in line_data["fontname"]
                    line_data["char_count"] = len(text)
                    line_data["word_count"] = len(text.split())
                    
                    all_lines_data.append(line_data)

        
        # (Keep your signature_frequency and page count logic down here)
        return all_lines_data
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        print(f"❌ PDF extraction error in {pdf_path} after {elapsed:.2f}s: {str(e)}")
        # Return empty list so the training/prediction loop can continue to the next file
        return []


# Extract text from PDF file
def parse_pdf(file_contents: bytes) -> str:
    try:
        pdf_file = BytesIO(file_contents)
        text_parts = []
        
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text and text.strip():
                    text_parts.append(text)
                    logger.debug(f"Extracted {len(text)} chars from page {page_num}")
        
        full_text = "\n\n".join(text_parts)
        logger.info(f"Successfully extracted {len(full_text)} characters from PDF")
        
        return full_text
    
    except Exception as e:
        logger.error(f"Failed to parse PDF: {str(e)}")
        raise ValueError(f"Could not parse PDF: {str(e)}")