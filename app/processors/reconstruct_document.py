import re
import json
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict
import uuid

def post_process_text(text: str) -> str:
    """
    Final text polish: Whitespace, Punctuation, Brackets, and Spacing.
    """
    if not text:
        return text

    #  Dash Normalization 
    text = re.sub(r'[\u2012\u2013\u2014\u2015]', '-', text)
    
    # Letter-Digit Separation 
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)

    # Punctuation Spacing
    text = re.sub(r'\s+([,.!?;:])', r'\1', text) # Remove space before
    text = re.sub(r'([,.!?;:])(?=[a-zA-Z])', r'\1 ', text) # Add space after if followed by char

    # Bracket / Parentheses Cleanup
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    text = re.sub(r'\[\s+', '[', text)
    text = re.sub(r'\s+\]', ']', text)

    # Whitespace Normalization
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def is_list_item(text: str) -> bool:
    """Check if a line is a list item (bullet, number, letter)."""
    patterns = [
        r'^[•\-*●○■□]',           # Bullets
        r'^\d+[\.)]\s',            # 1. or 1)
        r'^[a-z][\.)]\s',          # a. or a)
        r'^\([a-z\d]+\)\s',        # (a) or (1)
        r'^[IVXLCDM]+\.\s',        # Roman numerals
    ]
    return any(re.match(p, text.strip(), re.IGNORECASE) for p in patterns)


def should_merge_lines(prev_text, prev_bottom, curr_text, curr_top, curr_page, prev_page, font_size) -> bool:
    if curr_page != prev_page: return False
    if is_list_item(curr_text): return False
    if curr_text.isupper() and len(curr_text) > 3: return False
    
    v_gap = curr_top - prev_bottom
    TIGHT_THRESHOLD = font_size * 0.6
    LOOSE_THRESHOLD = font_size * 1.0
    
    if v_gap < TIGHT_THRESHOLD: return True
    if v_gap < LOOSE_THRESHOLD:
        return not prev_text.rstrip().endswith(('.', '!', '?', ':', ';'))
    return False


def reconstruct_document_json(labeled_lines, document_id=None, source_type="pdf", language="en") -> Dict[str, Any]:
    #SEPARATE CONTENT BY LABEL
    content_lines = []
    header_footer_lines = []
    title_lines = []
    
    for line in labeled_lines:
        label = int(line.get('label', 0))
        if label == 0: content_lines.append(line)
        elif label == 1: header_footer_lines.append(line)
        elif label == 2: title_lines.append(line)
    
    # BUILD SECTIONS WITH SMART PARAGRAPH GROUPING
    sections = []
    current_section = None
    all_lines_sorted = sorted(content_lines + title_lines, key=lambda x: (x.get('page_num', 0), x.get('y_pos_abs', 0)))
    prev_line_meta = None

    for line in all_lines_sorted:
        label = int(line.get('label', 0))
        text = line.get('text_content', '').strip()
        page_num = line.get('page_num', 1)
        y_top = line.get('y_pos_abs', 0)
        font_size = line.get('font_size', 10)
        y_bottom = line.get('bottom') or (y_top + (font_size * 1.2))

        if label == 2:  # TITLE
            if current_section: sections.append(current_section)
            current_section = {
                "id": f"sec_{len(sections) + 1}",
                "title": text,
                "level": 1,
                "page_start": page_num, "page_end": page_num,
                "content": [],
                "extraction_refs": {"emails": [], "phone_numbers": [], "dates": [], "currency": []}
            }
            prev_line_meta = None 

        elif label == 0:  # CONTENT
            if not current_section:
                current_section = {
                    "id": "sec_0", "title": "Preamble", "level": 1,
                    "page_start": page_num, "page_end": page_num,
                    "content": [], "extraction_refs": {"emails": [], "phone_numbers": [], "dates": [], "currency": []}
                }

            merge = False
            if prev_line_meta and current_section['content']:
                merge = should_merge_lines(prev_line_meta['text'], prev_line_meta['bottom'], text, y_top, page_num, prev_line_meta['page'], font_size)

            if merge:
                last_block = current_section['content'][-1]
                if last_block['text'].endswith('-'):
                    last_block['text'] = last_block['text'][:-1] + text
                else:
                    last_block['text'] += " " + text
            else:
                current_section['content'].append({
                    "id": f"blk_{len(current_section['content']) + 1}",
                    "type": "paragraph", "text": text, "page": page_num
                })
            
            current_section['page_end'] = page_num
            prev_line_meta = {"text": text, "bottom": y_bottom, "page": page_num}

    if current_section: sections.append(current_section)

    # APPLY POST-PROCESSING CLEANUP
    for section in sections:
        section['title'] = post_process_text(section['title'])
        for block in section['content']:
            block['text'] = post_process_text(block['text'])
    
    print(f"Stage 2: Created and post-processed {len(sections)} sections.")

    
    # EXTRACT ENTITIES (Now using cleaned text)
    extractions = {"emails": [], "phone_numbers": [], "dates": [], "currency": []}
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b'
    date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'
    currency_pattern = r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP)'

    extraction_counters = {"emails": 0, "phone_numbers": 0, "dates": 0, "currency": 0}

    for section in sections:
        for block in section['content']:
            text = block['text']
            
            for match in re.finditer(email_pattern, text, re.IGNORECASE):
                extraction_counters["emails"] += 1
                ext_id = f"email_{extraction_counters['emails']}"
                extractions["emails"].append({"id": ext_id, "value": match.group(0), "reliability": 0.95, "section_id": section['id'], "block_id": block['id'], "page": block['page'], "source": "regex"})
                section['extraction_refs']['emails'].append(ext_id)
            
            for match in re.finditer(phone_pattern, text):
                extraction_counters["phone_numbers"] += 1
                ext_id = f"phone_{extraction_counters['phone_numbers']}"
                extractions["phone_numbers"].append({"id": ext_id, "value": match.group(0), "reliability": 0.90, "section_id": section['id'], "block_id": block['id'], "page": block['page'], "source": "regex"})
                section['extraction_refs']['phone_numbers'].append(ext_id)

            for match in re.finditer(date_pattern, text, re.IGNORECASE):
                extraction_counters["dates"] += 1
                ext_id = f"date_{extraction_counters['dates']}"
                extractions["dates"].append({"id": ext_id, "value": match.group(0), "reliability": 0.85, "section_id": section['id'], "block_id": block['id'], "page": block['page'], "source": "regex"})
                section['extraction_refs']['dates'].append(ext_id)

            for match in re.finditer(currency_pattern, text, re.IGNORECASE):
                extraction_counters["currency"] += 1
                ext_id = f"currency_{extraction_counters['currency']}"
                extractions["currency"].append({"id": ext_id, "value": match.group(0), "reliability": 0.92, "section_id": section['id'], "block_id": block['id'], "page": block['page'], "source": "regex"})
                section['extraction_refs']['currency'].append(ext_id)

    
    # STAGE 4: PROCESS REMOVED CONTENT
    headers_by_text = defaultdict(list)
    footers_by_text = defaultdict(list)
    for line in header_footer_lines:
        t, y, p = line.get('text_content', '').strip(), line.get('y_pos_norm', 0.5), line.get('page_num', 1)
        if y < 0.15: headers_by_text[t].append(p)
        else: footers_by_text[t].append(p)

    removed_content = {
        "headers": [{"text": t, "pages": sorted(set(p)), "frequency": len(p)} for t, p in headers_by_text.items()],
        "footers": [{"text": t, "pages": sorted(set(p)), "frequency": len(p)} for t, p in footers_by_text.items()]
    }

    #ASSEMBLE FINAL JSON
    all_pages = [l.get('page_num', 1) for l in labeled_lines]
    
    return {
        "version": "1.0",
        "document_id": document_id or f"doc_{uuid.uuid4().hex[:8]}",
        "metadata": {
            "source_type": source_type, "page_count": max(all_pages) if all_pages else 0,
            "language": language, "processed_at": datetime.utcnow().isoformat() + "Z",
            "line_count": len(labeled_lines), "content_line_count": len(content_lines),
            "title_count": len(title_lines), "header_footer_count": len(header_footer_lines)
        },
        "sections": sections,
        "extractions": extractions,
        "removed": removed_content
    }

def save_reconstructed_json(json_data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved to: {output_path}")

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv('label_me_predicted.csv')
    res = reconstruct_document_json(df.to_dict('records'), document_id="training_data4.pdf")
    save_reconstructed_json(res, 'reconstructed_document_v2.json')