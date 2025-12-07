# document_processor
To run:
source venv/bin/activate
uvicorn app.main:app --reload 

Potential issues
remove_headers_footers treats all numbers on its own line within heading and footer lines as page numbers to be removed
is potential heading is way to aggressive. creating headings unintended