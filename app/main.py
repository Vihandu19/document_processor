import logging
import joblib
import pandas as pd
import json
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from typing import Literal

from app.processors.pdf_processor import extract_and_parse_pdf_linebyline
from app.processors.reconstruct_document import reconstruct_document_json
from app.models.response_models import DocumentResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document Processor API",
    description="Structural analysis and cleaning using Random Forest classification",
    version="1.0.0"
)

# Load the trained model and metadata once at startup
MODEL_PATH = "document_structure_model.pkl"
METADATA_PATH = "model_metadata.json"

model = None
required_features = []

try:
    # Load model
    model = joblib.load(MODEL_PATH)
    
    # Load required features from metadata (CRITICAL - don't hard-code!)
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
        required_features = metadata['model_features']
    
    logger.info(f"✓ Model loaded successfully with {len(required_features)} features")
    #logger.info(f"  Required features: {required_features}")
    
except FileNotFoundError as e:
    logger.error(f"CRITICAL: Model files not found: {e}")
    logger.error("Train a model first with: python train_model.py --train master_labeled_data.csv")
    
except Exception as e:
    logger.error(f"CRITICAL: Failed to load ML model: {e}")


@app.post("/process", response_model=DocumentResponse)
async def process_document(
    file: UploadFile = File(...),
    output_format: Literal["json", "markdown", "both"] = Query(default="json")
):
    """
    Complete Pipeline:
    1. Extraction (Geometry + Text)
    2. ML Classification (Identify Content/Title/Header-Footer)
    3. Reconstruction (Semantic JSON Building)
    """
    if not model:
        raise HTTPException(status_code=500, detail="ML model not initialized")

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # Step 1: Read file bytes
        contents = await file.read()
        file_size_mb = len(contents) / (1024 * 1024)
        
        # File size validation
        MAX_FILE_SIZE_MB = 10
        if file_size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large ({file_size_mb:.1f}MB). Maximum size is {MAX_FILE_SIZE_MB}MB"
            )
        
        logger.info(f"Processing {file.filename} ({file_size_mb:.2f}MB)")
        
        # Step 2: Line-by-Line Extraction
        lines_data = extract_and_parse_pdf_linebyline(contents)
        
        if not lines_data:
            raise HTTPException(
                status_code=422,
                detail="No text could be extracted from the PDF. The file may be empty, corrupted, or image-based."
            )
        
        logger.info(f"Extracted {len(lines_data)} lines from {file.filename}")

        # Step 3: Prepare Data for ML Prediction
        df = pd.DataFrame(lines_data)
        
        #  Ensure all required features exist Use the features loaded from model_metadata.json
        for feature in required_features:
            if feature not in df.columns:
                logger.warning(f"Missing feature '{feature}', filling with 0")
                df[feature] = 0
        
        # Extract only the features the model expects, in the correct order
        X = df[required_features].fillna(0)
        
        logger.info(f"Predicting labels for {len(X)} lines using {len(required_features)} features")
        
        # Predict labels (0=Content, 1=Header/Footer, 2=Title)
        predicted_labels = model.predict(X)
        df['label'] = predicted_labels
        
        # Log prediction distribution
        label_counts = pd.Series(predicted_labels).value_counts().sort_index()
        logger.info(f"Predictions - Content: {label_counts.get(0, 0)}, "
                   f"Headers/Footers: {label_counts.get(1, 0)}, "
                   f"Titles: {label_counts.get(2, 0)}")
        
        # Convert back to list of dicts for reconstruction
        labeled_lines = df.to_dict('records')

        # Step 4: Semantic Reconstruction
        logger.info(f"Reconstructing document structure")
        reconstructed_json = reconstruct_document_json(
            labeled_lines=labeled_lines,
            document_id=file.filename,
            source_type="pdf"
        )

        # Step 5: Generate Markdown if requested
        markdown_output = None
        if output_format in ["markdown", "both"]:
            markdown_output = convert_to_markdown(reconstructed_json)
            logger.info(f"Generated markdown ({len(markdown_output)} chars)")

        # Step 6: Build Response
        # Unpack the reconstructed_json dict into the response model
        response = DocumentResponse(
            filename=file.filename,
            status="success",
            # Unpack all the fields from reconstructed_json
            version=reconstructed_json["version"],
            document_id=reconstructed_json["document_id"],
            metadata=reconstructed_json["metadata"],
            sections=reconstructed_json["sections"],
            extractions=reconstructed_json["extractions"],
            removed=reconstructed_json["removed"],
            markdown=markdown_output
        )
        
        logger.info(f"✓ Successfully processed {file.filename}")
        return response

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
        
    except Exception as e:
        logger.exception(f"Error processing {file.filename}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal processing error: {str(e)}"
        )
        
    finally:
        await file.close()


def convert_to_markdown(structured_json: dict) -> str:
    """
    Convert structured JSON to clean markdown format.
    
    Args:
        structured_json: The reconstructed document JSON
        
    Returns:
        Markdown-formatted string
    """
    md_parts = []
    
    # Add document title if available
    doc_id = structured_json.get("document_id", "Document")
    md_parts.append(f"# {doc_id}\n")
    
    # Add metadata
    metadata = structured_json.get("metadata", {})
    if metadata:
        md_parts.append(f"**Pages:** {metadata.get('page_count', 'N/A')}  ")
        md_parts.append(f"**Language:** {metadata.get('language', 'N/A')}  ")
        md_parts.append(f"**Processed:** {metadata.get('processed_at', 'N/A')}\n")
    
    md_parts.append("\n---\n")
    
    # Add sections
    for section in structured_json.get("sections", []):
        title = section.get('title', 'Untitled')
        
        # Don't add header for "Preamble" section
        if title != "Preamble":
            # Use level for heading depth (## for level 1, ### for level 2, etc.)
            level = section.get('level', 1)
            heading_prefix = "#" * (level + 1)  # +1 because document title is #
            md_parts.append(f"\n{heading_prefix} {title}\n")
        
        # Add content blocks
        for block in section.get("content", []):
            text = block.get('text', '').strip()
            if text:
                md_parts.append(f"\n{text}\n")
    
    # Add extraction summary if there are extractions
    extractions = structured_json.get("extractions", {})
    total_extractions = sum(len(v) for v in extractions.values())
    
    if total_extractions > 0:
        md_parts.append("\n---\n")
        md_parts.append("\n## Extracted Information\n")
        
        if extractions.get('emails'):
            md_parts.append(f"\n**Emails:** {', '.join(e['value'] for e in extractions['emails'])}\n")
        
        if extractions.get('phone_numbers'):
            md_parts.append(f"\n**Phone Numbers:** {', '.join(p['value'] for p in extractions['phone_numbers'])}\n")
        
        if extractions.get('dates'):
            md_parts.append(f"\n**Dates:** {', '.join(d['value'] for d in extractions['dates'][:10])}")
            if len(extractions['dates']) > 10:
                md_parts.append(f" _(and {len(extractions['dates']) - 10} more)_")
            md_parts.append("\n")
        
        if extractions.get('currency'):
            md_parts.append(f"\n**Currency:** {', '.join(c['value'] for c in extractions['currency'][:10])}")
            if len(extractions['currency']) > 10:
                md_parts.append(f" _(and {len(extractions['currency']) - 10} more)_")
            md_parts.append("\n")
    
    return "".join(md_parts)


@app.get("/health")
async def health():
    """
    Health check endpoint.
    Returns system status and model information.
    """
    return {
        "status": "online",
        "model_loaded": model is not None,
        "features_count": len(required_features) if required_features else 0,
        "version": "1.0.0"
    }


@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "name": "Document Intelligence API",
        "version": "1.0.0",
        "description": "Structural analysis and cleaning using Random Forest classification",
        "endpoints": {
            "process": "/process (POST)",
            "health": "/health (GET)",
            "docs": "/docs (GET)"
        }
    }