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

#(V3) Extract text and features from PDF file line by line
def extract_and_parse_pdf_linebyline(
    file_bytes: bytes,
    x_tolerance: float = 8,
    y_tolerance: float = 5,
) -> List[Dict[str, Any]]:
    """
    Extract text-line features from a PDF using a hybrid approach:
    
    1. Line Extraction: Uses layout analysis to group text.
    2. Word Reconstruction: Uses individual words to fix spacing issues.
    3. Feature Engineering: Adds geometrical, repetition, and CONTEXTUAL features.
    
    Returns a list of feature dictionaries (one per line).
    """
    start_time = time.perf_counter()
    pdf_file = BytesIO(file_bytes)
    all_lines_data: List[Dict[str, Any]] = []

    try:
        with pdfplumber.open(pdf_file) as pdf:
            total_pages = len(pdf.pages)

            # --- PASS 1: EXTRACT LINES AND BASIC FEATURES ---
            for page_num, page in enumerate(pdf.pages):
                # Layout-based line detection
                lines = page.extract_text_lines(
                    layout=True,
                    use_text_flow=True,
                    x_tolerance=x_tolerance,
                    y_tolerance=y_tolerance,
                )

                # Word extraction for correct spacing reconstruction
                words = page.extract_words(
                    use_text_flow=True,
                    keep_blank_chars=False,
                    x_tolerance=1,
                    y_tolerance=3,
                )

                for line in lines:
                    # Helper: Reconstruct text from words to fix LaTeX spacing
                    text = reconstruct_line_from_words(line, words)
                    
                    # Fallback if reconstruction fails
                    if not text:
                        text = line.get("text", "").strip()

                    # Helper: Unicode normalization
                    text = pre_process_text(text)

                    # Skip empty or noise lines (e.g., single stray characters)
                    if not text or (len(text) == 1 and not text.isalnum()):
                        continue

                    # Geometry
                    x0, x1, top = line["x0"], line["x1"], line["top"]
                    
                    # Font extraction (Handle cases where chars might be missing)
                    chars = line.get("chars", [])
                    if chars:
                        # Use the most common font in the line to avoid outlier chars
                        # (e.g. a single bold char in a regular sentence)
                        fontname = chars[0].get("fontname", "UNKNOWN")
                        font_size = chars[0].get("size", 10.0)
                    else:
                        fontname = "UNKNOWN"
                        font_size = 10.0

                    # BASE FEATURES (Per-Line)
                    features = {
                        # Identifiers
                        "page_num": page_num + 1,
                        "line_signature": f"{round(top, 0)}_{fontname}_{round(font_size, 1)}",
                        "text_content": text,

                        # Geometry & Layout
                        "x0": x0,
                        "x1": x1,
                        "y_pos_abs": top,
                        "page_height": page.height,
                        "page_width": page.width,
                        "y_pos_norm": top / page.height,
                        "x_pos_norm": x0 / page.width,
                        "line_width_norm": (x1 - x0) / page.width,
                        "is_centered": abs((x0 + x1) / 2 - page.width / 2) < (page.width * 0.1),
                        
                        # Text Stats
                        "char_count": len(text),
                        "word_count": len(text.split()),
                        "is_short": len(text) < 30,
                        "is_very_short": len(text) < 10,
                        "is_single_word": len(text.split()) == 1,

                        # Typography
                        "fontname": fontname,
                        "font_size": font_size,
                        "is_bold": "bold" in fontname.lower(),
                        "is_italic": "italic" in fontname.lower(),
                        "is_all_caps": text.isupper() and len(text) > 2,
                        "is_title_case": text.istitle(),

                        # Regex Patterns
                        "is_numeric_pattern": bool(regex.match(r"^\s*[\(\[]?\s*\d+\s*[\)\]]?\s*$", text)),
                        "has_page_keyword": bool(regex.search(r"\bpage\b", text, regex.IGNORECASE)),
                        "has_date_pattern": bool(regex.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", text)),
                        "starts_with_number": bool(regex.match(r"^\d+[\.)]\s", text)),
                        "ends_with_colon": text.endswith(":"),

                        # Position
                        "is_top_10_percent": (top / page.height) < 0.1,
                        "is_bottom_10_percent": (top / page.height) > 0.9,
                        "is_first_page": page_num == 0,
                        "is_last_page": page_num == total_pages - 1,
                    }
                    all_lines_data.append(features)

            # count how many times exact line signatures appear across the doc
            signature_counts = {}
            for line in all_lines_data:
                sig = line["line_signature"]
                signature_counts[sig] = signature_counts.get(sig, 0) + 1

            for line in all_lines_data:
                sig = line["line_signature"]
                line["signature_frequency"] = signature_counts[sig]
                line["appears_on_multiple_pages"] = signature_counts[sig] > 1
                line["appears_on_most_pages"] = signature_counts[sig] >= (total_pages * 0.6)
            
            # Calculate global average font size to normalize "bigness"
            if all_lines_data:
                avg_font_size = sum(l['font_size'] for l in all_lines_data) / len(all_lines_data)
            else:
                avg_font_size = 10.0

            for i, line in enumerate(all_lines_data):
                # 1. Relative Font Size
                # (How much bigger is this than the document average?)
                line["font_size_rel"] = line["font_size"] / avg_font_size
                
                # Get neighbors (safely handling start/end of list)
                prev_line = all_lines_data[i - 1] if i > 0 else None
                next_line = all_lines_data[i + 1] if i < len(all_lines_data) - 1 else None
                
                # 2. Previous Line Signals
                if prev_line:
                    # Did the previous line look like the end of a sentence?
                    # If False, this line is likely a continuation (Label 0), not a Title (Label 2)
                    line["prev_ends_punctuation"] = prev_line["text_content"].strip().endswith(('.', '!', '?', ':'))
                    
                    # Formatting changes
                    line["font_size_change_prev"] = line["font_size"] - prev_line["font_size"]
                    line["is_different_style_prev"] = line["fontname"] != prev_line["fontname"]
                    
                    # Vertical distance (Handling page breaks)
                    if prev_line["page_num"] == line["page_num"]:
                        line["dist_from_prev"] = line["y_pos_abs"] - (prev_line["y_pos_abs"] + prev_line["font_size"])
                    else:
                        line["dist_from_prev"] = 100.0  # Large distance for new page
                else:
                    # Defaults for the very first line of the doc
                    line["prev_ends_punctuation"] = True
                    line["font_size_change_prev"] = 0
                    line["is_different_style_prev"] = False
                    line["dist_from_prev"] = 0

                # 3. Next Line Signals
                if next_line:
                    # Does the next line start with a lowercase?
                    # If True, this line is almost certainly NOT a Title (it's the start of a sentence)
                    cleaned_next = next_line["text_content"].strip()
                    line["next_starts_lower"] = cleaned_next[0].islower() if cleaned_next else False
                else:
                    line["next_starts_lower"] = False

        logger.info(f"PDF processed: {len(all_lines_data)} lines extracted in {time.perf_counter() - start_time:.2f}s")
        return all_lines_data

    except Exception as e:
        logger.exception("PDF extraction failed")
        # Return empty list so the pipeline doesn't crash completely
        return []


#(V2)extract text and features from PDF file
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


# (V2) Extract text from PDF file
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