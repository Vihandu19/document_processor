import re
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

# --- TEXT UTILITIES ---
def post_process_text(text: str) -> str:
    if not text: return ""
    text = re.sub(r'[\u2012\u2013\u2014\u2015]', '-', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def should_merge_lines(prev_line, curr_line) -> bool:
    # Logic from your original file
    if curr_line['page_num'] != prev_line['page_num']: return False
    v_gap = curr_line['y_pos_abs'] - prev_line['bottom']
    threshold = curr_line.get('font_size', 10) * 1.0
    return v_gap < threshold and not prev_line['text_content'].strip().endswith(('.', '!', '?', ':'))

# --- HIERARCHY ENGINE ---
class HierarchyBuilder:
    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.chunks = []
        self.stack = [{"title": "DOCUMENT_ROOT", "level": 0, "font_size": 999, "path": []}]
        self.current_para = None

    def process_line(self, line: Dict):
        label = int(line.get('label', 0))
        text = line.get('text_content', '').strip()
        f_size = line.get('font_size', 10)
        
        if not text: return

        if label == 2:  # TITLE (Heading)
            # Find parent by font size
            while len(self.stack) > 1 and f_size >= (self.stack[-1]['font_size'] - 0.5):
                self.stack.pop()
            
            new_path = self.stack[-1]['path'] + [text]
            self.stack.append({
                "title": text, 
                "level": len(self.stack), 
                "font_size": f_size, 
                "path": new_path
            })
            self.current_para = None # New section breaks paragraph flow

        elif label == 0:  # CONTENT
            if self.current_para and should_merge_lines(self.current_para['_last_line'], line):
                self.current_para['text'] += " " + text
                self.current_para['line_refs'].append(line.get('line_id', 0))
                self.current_para['_last_line'] = line
            else:
                self.create_chunk()
                self.current_para = {
                    "text": text,
                    "page": line['page_num'],
                    "line_refs": [line.get('line_id', 0)],
                    "_last_line": line
                }

    def create_chunk(self):
        if not self.current_para: return
        
        active_section = self.stack[-1]
        raw_text = self.current_para['text']
        
        # Entity Extraction per chunk
        amounts = [float(s.replace(',', '')) for s in re.findall(r'\d+(?:,\d{3})*(?:\.\d{2})?', raw_text)]
        
        self.chunks.append({
            "chunk_id": f"{self.doc_id}_sec{len(self.stack)}_p{len(self.chunks)}",
            "text": post_process_text(raw_text),
            "metadata": {
                "document_id": self.doc_id,
                "section_title": active_section['title'],
                "section_path": active_section['path'],
                "currency": re.findall(r'USD|EUR|GBP|\$', raw_text),
                "amounts": amounts,
                "Dates": re.findall(r'\b\d{4}-\d{2}-\d{2}\b', raw_text),
                "document_type": "contract" # Or dynamic from doc metadata
            },
            "anchors": {
                "page_range": [self.current_para['page'], self.current_para['page']],
                "line_refs": self.current_para['line_refs']
            }
        })

def reconstruct_document_json(labeled_lines, document_id=None):
    builder = HierarchyBuilder(document_id or str(uuid.uuid4()))
    
    # Sort lines
    sorted_lines = sorted(labeled_lines, key=lambda x: (x['page_num'], x['y_pos_abs']))
    
    for line in sorted_lines:
        # Calculate 'bottom' if missing for merge logic
        if 'bottom' not in line:
            line['bottom'] = line['y_pos_abs'] + line.get('font_size', 10)
        builder.process_line(line)
    
    builder.create_chunk() # Flush last paragraph
    return {"chunks": builder.chunks}