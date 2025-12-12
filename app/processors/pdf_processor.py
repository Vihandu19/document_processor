import pypdf
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

#Extract text from PDF file
async def parse_pdf(file_contents: bytes) -> str:
    try:
        # Create a file-like object
        pdf_file = BytesIO(file_contents)
        
        # Create PDF reader
        reader = pypdf.PdfReader(pdf_file)
        
        # Extract text from all pages
        text_parts = []
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text.strip():  # Only add non-empty pages
                text_parts.append(text)
                logger.debug(f"Extracted {len(text)} chars from page {page_num}")
        
        full_text = "\n\n".join(text_parts)
        logger.info(f"Successfully extracted {len(full_text)} characters from PDF")
        
        return full_text
        
    except Exception as e:
        logger.error(f"Failed to parse PDF: {str(e)}")
        raise ValueError(f"Could not parse PDF: {str(e)}")