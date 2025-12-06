from pydantic import BaseModel
from typing import Optional, Dict, Any

class DocumentResponse(BaseModel):
    filename: str
    status: str
    markdown: Optional[str] = None
    json_data: Optional[Dict[str, Any]] = None