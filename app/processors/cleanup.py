import logging
import time
from typing import Dict, Any, List

import regex


logger = logging.getLogger(__name__)


#reconstruct text from line-level features
def reconstruct_text_from_features(lines: List[Dict[str, Any]],
    remove_headers: bool = False) -> str:
    """
    Rebuilds text from line-level PDF features.
    Removes repeated headers/footers using signature frequency.
    """
    start_time = time.perf_counter()
    if not lines:
        return ""

    # Count signature frequencies to detect repeated header/footer lines
    signature_counts = {}
    for ln in lines:
        sig = ln["line_signature"]
        signature_counts[sig] = signature_counts.get(sig, 0) + 1

    # Repeated lines likely headers/footers (appear on most pages)
    total_pages = max(ln["page_num"] for ln in lines)
    threshold = max(2, total_pages * 0.6)   # appears in 60%+ of pages

    header_footer_signatures = {
        sig for sig, count in signature_counts.items() if count >= threshold
    } if remove_headers else set() 

    # Sort lines top->bottom and left->right
    lines_sorted = sorted(
        lines,
        key=lambda ln: (ln["page_num"], ln["y_pos_abs"], ln["x0"])
    )

    # Build output text while skipping H/F
    output_lines = []
    for ln in lines_sorted:
        if ln["line_signature"] in header_footer_signatures:
            continue
        output_lines.append(ln["text_content"])

    return "\n".join(output_lines)


async def clean_text(raw_text: str) -> Dict[str, Any]:
    """
    Alternative clean_text function that uses reconstruct_text_from_features.
    """
    try:
        text = normalize_newlines(raw_text)
        text = fix_spacing(text)
        sections = identify_sections(text)
        markdown = convert_to_markdown(sections)
        if not sections:
            sections = [{"heading": "Content", "content": text.strip()}]
        json_data = create_json(sections, text)

        end_time = time.perf_counter()
        logger.info(f"Successfully cleaned and structured text (alt method) in {end_time:.3f}s")

        return {
            "markdown": markdown,
            "json": json_data
        }
    except Exception as e:
        end_time = time.perf_counter()
        logger.error(f"Failed to clean text (alt method)in {end_time:.3f}: {str(e)}")
        raise ValueError(f"Could not clean text: {str(e)}")


#Normalize newlines and intelligently join/split lines
def normalize_newlines(text: str) -> str:

    known_headings = {
        "Introduction", "Conclusion", "Summary", "Abstract", 
        "References", "Discussion", "Methodology", "Results", 
        "Findings", "Background", "Group Dynamics", "Reflection",
        "Objectives", "Scope", "Limitations", "Recommendations", 
        "Appendix", "Acknowledgements", "Future Work", "Literature Review"
    }
    
    #Normalize spaces 
    text = regex.sub(r'  +', ' ', text)
    
    #remove lines that are just whitespace
    lines = [line.strip() for line in text.split('\n')]
    
    # Filter out completely empty lines for now
    non_empty_lines = [line for line in lines if line]
    
    logger.info(f"After space normalization: {len(non_empty_lines)} non-empty lines")
    
    result = []
    i = 0
    
    while i < len(non_empty_lines):
        line = non_empty_lines[i]
        
        # Check if this is a known heading - keep separate
        if line in known_headings:
            logger.info(f"Found known heading: {line}")
            result.append("")  # Blank line before heading
            result.append(line)
            result.append("")  # Blank line after heading
            i += 1
            continue
        
        # Start building a paragraph
        paragraph = [line]
        i += 1
        
        # Keep joining until we hit a heading or end
        while i < len(non_empty_lines):
            next_line = non_empty_lines[i]
            
            # Stop at known heading
            if next_line in known_headings:
                break
            
            # Stop at ALL CAPS (likely heading) - must be >2 chars
            if next_line.isupper() and len(next_line) > 3 and ' ' in next_line:
                break
            
            # Stop at numbered heading (1., 1.2, etc.)
            if regex.match(r'^\d+(\.\d+)*[\.)]\s+[A-Z]', next_line):
                break
            
            # Otherwise, keep joining
            paragraph.append(next_line)
            i += 1
        
        # Join paragraph with spaces
        joined = ' '.join(paragraph)
        result.append(joined)
    
    # Join with double newlines (paragraph breaks)
    text = '\n\n'.join(result)
    
    # Clean up excessive blank lines (3+ → 2)
    text = regex.sub(r'\n{3,}', '\n\n', text)
    
    #logger.info(f"normalize_newlines OUTPUT - First 300 chars: {text[:300]}")
    #logger.info(f"normalize_newlines OUTPUT - Newline count: {text.count(chr(10))}")
    
    return text


#Remove headers and footers including dates and page numbers
def remove_headers_footers(text: str, header_lines: int = 2, footer_lines: int = 2) -> str:   
    date_patterns = [
    r'\b\d{1,2}/\d{1,2}/\d{4}\b',   # MM/DD/YYYY or DD/MM/YYYY
    r'\b\d{1,2}-\d{1,2}-\d{4}\b',   # MM-DD-YYYY or DD-MM-YYYY
    r'\b\d{1,2}/\d{1,2}/\d{2}\b',   # MM/DD/YY or DD/MM/YY
    r'\b\d{1,2}-\d{1,2}-\d{2}\b',   # MM-DD-YY or DD-MM-YY
    ]

    page_patterns = [
    r'\bPage\s+\d+\b',              # "Page 5"
    r'\b\d+\s+of\s+\d+\b',          # "5 of 10"
    r'\bPage\s+\d+\s+of\s+\d+\b',   # "Page 5 of 10"
    r'^\s*\d+\s*$',                 #  5 (standalone number - possibe issue)
    r'\[\d+\]',                     # [5]
    r'\(\d+\)',                     # (5)
    ] 

    lines = text.split("\n")
    cleaned_lines = []

    #remove dates from headers/footers
    for i, line in enumerate(lines):
        if i < header_lines or i >= len(lines) - footer_lines:
            for pattern in date_patterns:
                line = regex.sub(pattern, '', line, flags=regex.IGNORECASE)
            for pattern in page_patterns:
                line = regex.sub(pattern, '', line, flags=regex.IGNORECASE)
            line = line.strip()  # remove leftover spaces
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    return text


#fix spacing issues without destroying formatting, tables, or real hyphens.
def fix_spacing(text: str) -> str:
    # Replace multiple empty lines with exactly 2 newlines
    text = regex.sub(r'\n{3,}', '\n\n', text)
    
    # Remove trailing spaces at the end of lines
    text = regex.sub(r'[ \t]+$', '', text, flags=regex.MULTILINE)
    
    # Fix hyphenated words split across lines
    text = regex.sub(r'([A-Za-z]{2,})-\n([a-z]{2,})', r'\1\2', text)
    
    # Normalize spaces on all lines (except empty lines)
    cleaned_lines = []
    for line in text.split("\n"):
        if line.strip():  # non-empty line
            # Replace 2+ spaces with a single space
            line = regex.sub(r' {2,}', ' ', line)
        cleaned_lines.append(line)
    
    return "\n".join(cleaned_lines).strip()



#Determine if a line is likely a section heading using a scoring system
def is_potential_heading(line: str, prev_blank: bool, next_blank: bool) -> bool:
    known_headings = ["Reflection", "Group Dynamics", "Conclusion", "Introduction",
                      "Abstract", "Summary", "Results", "Discussion",
                      "Methodology", "Literature Review", "Findings",
                      "Recommendations", "Appendix", "References",
                      "Acknowledgements", "Background", "Objectives",
                      "Scope", "Limitations", "Future Work"]
    feild_labels = {
    "total", "amount", "email", "address", "phone", "date", "invoice",
    "subtotal", "tax", "balance", "signature", "name", "price", "qty",
    "quantity", "description", "item", "payment", "due", "from", "to",
    "subject", "re", "cc", "bcc", "attn", "attention"
    }
    score = 0
    stripped = line.strip()

    #check against known headings
    for x in known_headings:
        if stripped.startswith(x):
            return True
        
    #must have some content
    if len(stripped) < 2:
        return False
    
    #reject all feild labels
    if stripped.lower().rstrip(':') in feild_labels:
        return False
    
    #reject lines in the format "label: value"
    if regex.match(r'^[A-Za-z\s]+:\s+[\w\d@.]', stripped):
        return False

    #reject bullet points
    if regex.match(r'^[-•*]\s+', stripped):
        return False
    
    #reject table like rows
    if regex.search(r'\S\s{2,}\S', stripped):
        return False
    if '|' in stripped and stripped.count('|') >= 2:
        return False
    
    #increase score for short all caps line
    if stripped.isupper() and 3 <= len(stripped) <= 80:
        score += 3
    
    # ritle Case short phrases (only if context suggests heading)
    if len(stripped.split()) <= 4 and stripped.istitle() and prev_blank:
        score += 2
    
    # numbered headings (1. or 1.2 or 1.2.3)
    if regex.match(r'^\d+(\.\d+)*[\.)]\s+[A-Z]', stripped):
        score += 3
    
    #roman numeral headings (I., II., III., IV., etc.)
    if regex.match(r'^[IVXLCDM]+\.\s+[A-Z]', stripped):
        score += 3
    
    #letter headings (A., B., C.)
    if regex.match(r'^[A-Z]\.\s+', stripped) and len(stripped.split()) <= 8:
        score += 2
    
    #lines ending with ":" that look like headings
    if stripped.endswith(':') and len(stripped.split()) <= 8 and not any(char.isdigit() for char in stripped):
        score += 2
    
    #surrounded by blank lines
    if prev_blank and next_blank:
        score += 3
    
    #only previous blank (common pattern)
    elif prev_blank:
        score += 1
    
    #short line
    if len(stripped) <= 60:
        score += 1
    
    return score >= 6


#Identify document sections and headings using  pattern matching
def identify_sections(text: str) -> List[Dict[str, str]]:
    lines = text.split('\n')
    sections = []
    current_section = {"heading": None, "content": []}
    
    for i, raw_line in enumerate(lines):
        line = raw_line.rstrip()  # Preserve indentation on left side
        
        if not line.strip():  
            current_section["content"].append("")
            continue
        
        prev_blank = (i > 0 and not lines[i - 1].strip())
        next_blank = (i + 1 < len(lines) and not lines[i + 1].strip())
        
        if is_potential_heading(line, prev_blank, next_blank):
            # Save current section if it has content
            content_text = "\n".join(current_section["content"]).strip()
            if content_text:
                sections.append({
                    "heading": current_section["heading"] or "Introduction",
                    "content": content_text
                })
            
            # Start new section
            heading = line.strip().rstrip(':')
            current_section = {"heading": heading, "content": []}
        else:
            current_section["content"].append(raw_line)
    
    # Finalize last section
    final_text = "\n".join(current_section["content"]).strip()
    if final_text:
        sections.append({
            "heading": current_section["heading"] or "Content",
            "content": final_text
        })
    
    return sections


#convert structured sections to markdown format    
def convert_to_markdown(sections: List[Dict[str, str]]) -> str:
    markdown_parts = []
    
    for section in sections:
        heading = section.get("heading", "Section")
        content = section.get("content", "")
        
        # Add heading with proper markdown syntax
        markdown_parts.append(f"## {heading}\n")
        
        # Add content with proper spacing
        markdown_parts.append(f"{content}\n")
    
    return "\n".join(markdown_parts).strip()


#Create a JSON representation of the document structure
def create_json(sections: List[Dict[str, str]], full_text: str) -> Dict[str, Any]:
    # Extract metadata
    word_count = len(full_text.split())
    char_count = len(full_text)
    
    # Detect email and phone patterns
    emails = regex.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', full_text)
    phones = regex.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', full_text)
    
    # Detect dates in various formats
    dates = []
    date_patterns = [
        r'\b\d{1,2}/\d{1,2}/\d{4}\b',   # MM/DD/YYYY
        r'\b\d{1,2}-\d{1,2}-\d{4}\b',   # MM-DD-YYYY
        r'\b\d{1,2}/\d{1,2}/\d{2}\b',   # MM/DD/YY
        r'\b\d{1,2}-\d{1,2}-\d{2}\b',   # MM-DD-YY
    ]
    for pattern in date_patterns:
        dates.extend(regex.findall(pattern, full_text))

    # Build JSON structure correctly
    json_data = {
        "sections": [
            {
                "heading": section.get("heading", "Untitled"),
                "content": section.get("content", ""),
                "word_count": len(section.get("content", "").split())
            }
            for section in sections
        ],
        "metadata": {
            "total_sections": len(sections),
            "word_count": word_count,
            "character_count": char_count,
            "emails_found": list(set(emails)),
            "phones_found": list(set(phones)),
            "dates_found": list(set(dates))[:10]
        },
    }

    return json_data

    