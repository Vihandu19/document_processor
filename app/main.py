from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
import logging
from typing import Literal
from app.models.response_models import DocumentResponse
from app.processors.pdf_processor import parse_pdf
from app.processors.docx_processor import parse_docx
from app.processors.cleanup import clean_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document Cleanup API",
    description="Clean and structure messy PDFs and DOCX files",
    version="1.0.0"
)

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB



@app.post("/process", response_model=DocumentResponse)
async def process_document(
    file: UploadFile = File(...),
    output_format: Literal["markdown", "json", "both"] = Query(
        default="markdown",
        description="Output format preference"
    )):

    """
    Process and clean a document (PDF or DOCX)
    input_formate: Upload PDF or DOCX file
    output_format: Choose 'markdown', 'json', or 'both'
    # Validate file type
    """""
    
    allowed_types = ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Only PDF and DOCX allowed."
        )
    
    try:
        # Read file contents
        contents = await file.read()
        #valid file size
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024}MB"
            )
        
        #process pdf
        if file.content_type == "application/pdf":
            logger.info(f"Processing PDF: {file.filename}")
            raw_text = await parse_pdf(contents)

            # DEBUG: Log raw PDF extraction
            logger.info(f"RAW PDF TEXT - First 300 chars: {raw_text[:300]}")
            logger.info(f"RAW PDF TEXT - Newline count: {raw_text.count(chr(10))}")
            logger.info(f"RAW PDF TEXT - Space count: {raw_text.count(' ')}")

        #process docx
        else:  
            logger.info(f"Processing DOCX: {file.filename}")
            raw_text = await parse_docx(contents)
        
        # Clean and structure the text
        cleaned = await clean_text(raw_text)

        sections = cleaned.get("json", {}).get("sections", [])
        logger.info(f"Found {len(sections)} sections: {[s['heading'] for s in sections]}")

        # Build response based on requested format
        response_data = {
            "filename": file.filename,
            "status": "success",
            "markdown": "",     
            "json_data": {}   
        }
        if output_format in ["markdown", "both"]:
            response_data["markdown"] = cleaned["markdown"]
        if output_format in ["json", "both"]:
            response_data["json_data"] = cleaned["json"]
        
        return DocumentResponse(**response_data)
        
    #error handling    
    except Exception as e:
        logger.error(f"Error processing {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {str(e)}"
        )
    
    #Close file after processing
    finally:
        await file.close()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "document-cleanup-api"}