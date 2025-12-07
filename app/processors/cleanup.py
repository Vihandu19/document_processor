import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

FIELD_LABELS = {
    "total", "amount", "email", "address", "phone", "date", "invoice",
    "subtotal", "tax", "balance", "signature", "name", "price", "qty",
    "quantity", "description", "item", "payment", "due", "from", "to",
    "subject", "re", "cc", "bcc", "attn", "attention"
    }

#Clean and structure raw text extracted from documents
#Args: raw_text: The unprocessed text extracted from a document
#Returns: a dictionary with cleaned markdown and structured JSON data
async def clean_text(raw_text: str) -> Dict[str, Any]:
    try:
        #Normalize newlines from pdf -> plaintext extraction
        text = normalize_newlines(raw_text)

        #Remove common headers/footers patterns
        text = remove_headers_footers(text)
        
        #Fix spacing and indentation
        text = fix_spacing(text)
        
        #Identify structure (sections, headings)
        sections = identify_sections(text)
        
        #Convert to markdown
        markdown = convert_to_markdown(sections)
        if not sections:
            sections = [{"heading": "Content", "content": text.strip()}]

        #Create JSON structure
        json_data = create_json(sections, text)
        
        logger.info("Successfully cleaned and structured text")
        
        return {
            "markdown": markdown,
            "json": json_data
        }
    
    except Exception as e:
        logger.error(f"Failed to clean text: {str(e)}")
        raise ValueError(f"Could not clean text: {str(e)}")
    

#fix spurious line breaks in PDF text.
#replace single newlines inside paragraphs with spaces. Keep double newlines as real paragraph breaks.
def normalize_newlines(text: str) -> str:

    #replace single newlines with space
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    #normalize multiple empty lines to double newline
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text


#Remove headers and footers from the text
#Args:
    #text: The raw text extracted from the document
    #header_lines: Number of lines to consider as header (default 1)
    #footer_lines: Number of lines to consider as footer (default 1)
#Returns: 
#   Cleaned text without headers and footers
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
                line = re.sub(pattern, '', line, flags=re.IGNORECASE)
            for pattern in page_patterns:
                line = re.sub(pattern, '', line, flags=re.IGNORECASE)
            line = line.strip()  # remove leftover spaces
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    return text


#fix spacing issues without destroying formatting, tables, or real hyphens.
#Args: text: The raw text extracted from the document
#Returns: Cleaned text without headers and footers
def fix_spacing(text: str) -> str:
    # Replace multiple empty lines with exactly 2 newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove trailing spaces at the end of lines
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    
    # Fix hyphenated words split across lines
    text = re.sub(r'([A-Za-z]{2,})-\n([a-z]{2,})', r'\1\2', text)
    
    # Normalize spaces on all lines (except empty lines)
    cleaned_lines = []
    for line in text.split("\n"):
        if line.strip():  # non-empty line
            # Replace 2+ spaces with a single space
            line = re.sub(r' {2,}', ' ', line)
        cleaned_lines.append(line)
    
    return "\n".join(cleaned_lines).strip()



#Determine if a line is likely a section heading using a scoring system
#Args:
#   line: The line to evaluate
#   prev_blank: Whether the previous line was blank
#   next_blank: Whether the next line is blank
#Returns:
    #True if the line is likely a heading

def is_potential_heading(line: str, prev_blank: bool, next_blank: bool) -> bool:
    score = 0
    stripped = line.strip()

    #must have some content
    if len(stripped) < 2:
        return False
    
    #reject all feild labels
    if stripped.lower().rstrip(':') in FIELD_LABELS:
        return False
    
    #reject lines in the format "label: value"
    if re.match(r'^[A-Za-z\s]+:\s+[\w\d@.]', stripped):
        return False

    #reject bullet points
    if re.match(r'^[-•*]\s+', stripped):
        return False
    
    #reject table like rows
    if re.search(r'\S\s{2,}\S', stripped):
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
    if re.match(r'^\d+(\.\d+)*[\.)]\s+[A-Z]', stripped):
        score += 3
    
    #roman numeral headings (I., II., III., IV., etc.)
    if re.match(r'^[IVXLCDM]+\.\s+[A-Z]', stripped):
        score += 3
    
    #letter headings (A., B., C.)
    if re.match(r'^[A-Z]\.\s+', stripped) and len(stripped.split()) <= 8:
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
    
    #short line (common for headings)
    if len(stripped) <= 60:
        score += 1
    
    #return true if score meets threshold
    return score >= 7


#Identify document sections and headings using intelligent pattern matching
#Args: text: The cleaned text to analyze
#Returns: List of sections with headings and content
def identify_sections(text: str) -> List[Dict[str, str]]:
    lines = text.split('\n')
    sections = []
    current_section = {"heading": None, "content": []}
    
    for i, raw_line in enumerate(lines):
        line = raw_line.rstrip()  # Preserve indentation on left side
        
        if not line.strip():  # Empty line
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
# Args: sections: List of sections with headings and content
# Returns: markdown-formatted text
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
#Args:
#   sections: List of sections with headings and content
#   full_text: The complete cleaned text
#Returns:
#   Structured JSON representation with metadata
def create_json(sections: List[Dict[str, str]], full_text: str) -> Dict[str, Any]:
    # Extract metadata
    word_count = len(full_text.split())
    char_count = len(full_text)
    
    # Detect email and phone patterns
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', full_text)
    phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', full_text)
    
    # Detect dates in various formats
    dates = []
    date_patterns = [
        r'\b\d{1,2}/\d{1,2}/\d{4}\b',   # MM/DD/YYYY
        r'\b\d{1,2}-\d{1,2}-\d{4}\b',   # MM-DD-YYYY
        r'\b\d{1,2}/\d{1,2}/\d{2}\b',   # MM/DD/YY
        r'\b\d{1,2}-\d{1,2}-\d{2}\b',   # MM-DD-YY
    ]
    for pattern in date_patterns:
        dates.extend(re.findall(pattern, full_text))

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
        "summary": sections[0].get("content", "")[:200] + "..."
        if sections and sections[0].get("content")
        else "No content available"
    }

    return json_data

    