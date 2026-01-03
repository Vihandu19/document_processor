import logging
import joblib
import pandas as pd
import json
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from typing import Literal, List

from app.processors.pdf_processor import extract_and_parse_pdf_linebyline
from app.processors.reconstruct_document import reconstruct_document_json
# Assuming you update your response models file too (see step 2)
from app.models.response_models import DocumentResponse 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Document Intelligence API", version="1.1.0")

MODEL_PATH = "document_structure_model.pkl"
METADATA_PATH = "model_metadata.json"

model = None
required_features = []

try:
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
        required_features = metadata['model_features']
    logger.info(f"✓ Model loaded with {len(required_features)} features")
except Exception as e:
    logger.error(f"CRITICAL: Failed to load ML model: {e}")

@app.post("/process", response_model=DocumentResponse)
async def process_document(
    file: UploadFile = File(...),
    output_format: Literal["json", "markdown", "both"] = Query(default="json")
):
    if not model:
        raise HTTPException(status_code=500, detail="ML model not initialized")

    try:
        contents = await file.read()
        
        # 1. Extraction (Now includes your contextual features)
        lines_data = extract_and_parse_pdf_linebyline(contents)
        if not lines_data:
            raise HTTPException(status_code=422, detail="No text extracted")

        # 2. ML Prediction
        df = pd.DataFrame(lines_data)
        for feature in required_features:
            if feature not in df.columns:
                df[feature] = 0
        
        X = df[required_features].fillna(0)
        df['label'] = model.predict(X)
        
        # 3. Hierarchical Reconstruction
        # This now returns the 'chunks' format we built earlier
        reconstructed_json = reconstruct_document_json(
            labeled_lines=df.to_dict('records'),
            document_id=file.filename
        )

        # 4. Generate Markdown from Chunks
        markdown_output = None
        if output_format in ["markdown", "both"]:
            markdown_output = convert_chunks_to_markdown(reconstructed_json["chunks"])

        # 5. Build Response
        return {
            "document_id": file.filename,
            "status": "success",
            "chunks": reconstructed_json["chunks"]
        }
    
    except Exception as e:
        logger.exception(f"Error processing {file.filename}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await file.close()

def convert_chunks_to_markdown(chunks: List[dict]) -> str:
    """
    Intelligently reconstructs Markdown by looking at the 'section_path'.
    If the path changes between chunks, it inserts the appropriate headings.
    """
    md_parts = []
    last_path = []

    for chunk in chunks:
        current_path = chunk["metadata"].get("section_path", [])
        
        # If the section path has changed, find where it diverged
        if current_path != last_path:
            for i, section_title in enumerate(current_path):
                # If this part of the path is new, print it as a header
                if i >= len(last_path) or section_title != last_path[i]:
                    level = i + 1
                    prefix = "#" * (level + 1) # ## for H1, ### for H2
                    md_parts.append(f"\n{prefix} {section_title}\n")
            
            last_path = current_path

        # Add the paragraph text
        text = chunk.get("text", "").strip()
        if text:
            md_parts.append(f"\n{text}\n")

    return "".join(md_parts)