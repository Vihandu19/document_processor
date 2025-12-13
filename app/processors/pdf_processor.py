import pdfplumber
import regex
import time
import logging

from typing import List, Dict, Any
from itertools import groupby
from io import BytesIO

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
    Enhanced with ML-ready features for header/footer/title detection.
    """
    start_time = time.perf_counter()
    pdf_file = BytesIO(file_bytes)
    all_lines_data = []

    try:
        with pdfplumber.open(pdf_file) as pdf:
            total_pages = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages):
                words = page.extract_words(
                    x_tolerance=3,
                    y_tolerance=3,
                    extra_attrs=["fontname", "size"]
                )

                if not words:
                    continue

                line_key = lambda w: (
                    round(w["top"], 1),
                    w.get("fontname", "UNKNOWN"),
                    round(w.get("size", 0), 1)
                )

                words = sorted(words, key=line_key)

                for signature, group in groupby(words, key=line_key):
                    line_words = list(group)

                    x0 = min(w["x0"] for w in line_words)
                    x1 = max(w["x1"] for w in line_words)
                    top = min(w["top"] for w in line_words)
                    text = " ".join(w["text"] for w in line_words)
                    text_stripped = text.strip()

                    #eatures for ML
                    features = {
                        #identifiers
                        "page_num": page_num + 1,
                        "line_signature": str(signature),
                        "text_content": text,
                        
                        #layeout features
                        "x0": x0,
                        "x1": x1,
                        "y_pos_abs": top,
                        "page_height": page.height,
                        "page_width": page.width,
                        "y_pos_norm": top / page.height,  # 0=top, 1=bottom
                        "x_pos_norm": x0 / page.width,    # 0=left, 1=right
                        "line_width": x1 - x0,
                        "line_width_norm": (x1 - x0) / page.width,
                        "is_centered": abs((x0 + x1) / 2 - page.width / 2) < (page.width * 0.1),
                        
                        #text features
                        "char_count": len(text_stripped),
                        "word_count": len(text_stripped.split()),
                        "is_short": len(text_stripped) < 30,
                        "is_very_short": len(text_stripped) < 10,
                        "is_single_word": len(text_stripped.split()) == 1,
                        
                        #typographic features
                        "fontname": signature[1],
                        "font_size": signature[2],
                        "is_bold": "bold" in signature[1].lower(),
                        "is_italic": "italic" in signature[1].lower(),
                        
                        #content features
                        "is_all_caps": text_stripped.isupper() and len(text_stripped) > 2,
                        "is_title_case": text_stripped.istitle(),
                        "is_numeric_pattern": bool(regex.match(r'^\s*[\(\[]?\s*\d+\s*[\)\]]?\s*$', text_stripped)),
                        "has_page_keyword": bool(regex.search(r'\bpage\b', text_stripped, regex.IGNORECASE)),
                        "has_date_pattern": bool(regex.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text_stripped)),
                        "starts_with_number": bool(regex.match(r'^\d+[\.)]\s', text_stripped)),
                        "ends_with_colon": text_stripped.endswith(':'),
                        
                        #positional features
                        "is_top_10_percent": (top / page.height) < 0.1,
                        "is_bottom_10_percent": (top / page.height) > 0.9,
                        "is_first_page": page_num == 0,
                        "is_last_page": page_num == total_pages - 1,
                    }

                    all_lines_data.append(features)

            #post processing
            # Count how many times each line signature appears
            signature_counts = {}
            for line in all_lines_data:
                sig = line["line_signature"]
                signature_counts[sig] = signature_counts.get(sig, 0) + 1
            
            # Add repetition features
            for line in all_lines_data:
                sig = line["line_signature"]
                line["signature_frequency"] = signature_counts[sig]
                line["appears_on_multiple_pages"] = signature_counts[sig] > 1
                line["appears_on_most_pages"] = signature_counts[sig] >= (total_pages * 0.6)

        end_time = time.perf_counter()
        logger.debug(f"PDF Extracted in  {end_time - start_time:.2f} seconds")
        return all_lines_data

    except Exception as e:
        logger.error(f"PDF extraction error after {end_time - start_time:.2f} seconds: {e}")
        return []