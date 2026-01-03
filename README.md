# document_processor

This project provides an end-to-end solution for transforming unstructured or messy documents (specifically PDFs and DOCX files) into clean, structured data formats (JSON and Markdown) suitable for downstream applications such as search indexing, data extraction, and analysis.

The core innovation is the use of a Machine Learning (ML) classifier to accurately determine the structural role of every line of text. It distinguishes essential content (body text, headings) from repetitive noise (e.g., headers, footers, and page numbers), allowing for intelligent content cleanup.

## Sample Output: List of chunks in this format 

{
  "document_id": "service_agreement_v2.pdf",
  "status": "success",
  "chunks": [
    {
      "chunk_id": "service_agreement_v2.pdf_sec2_p0",
      "text": "The Consultant shall be paid a total fee of 5,500.00 USD for the services provided. This payment is due no later than 2024-12-31 and includes all applicable taxes and administrative costs.",
      "metadata": {
        "document_id": "service_agreement_v2.pdf",
        "section_title": "3. Payment Terms",
        "section_path": [
          "EXHIBIT A: STATEMENT OF WORK",
          "3. Payment Terms"
        ],
        "currency": ["USD"],
        "amounts": [5500.0),
        "Dates": ["2024-12-31"],
        "document_type": "contract"
      },
      "anchors": {
        "page_range": [3, 3],
        "line_refs": [442, 443, 444]
      }
    }
  ]
}

## Restrictions

* Text box based files
* Scanned or image-only PDFs are not supported

## Prerequisites

* Python 3.8+

## Environment Activation

```bash
source venv/bin/activate
```

## Service Startup (API)

```bash
uvicorn app.main:app --reload
```

* The API will be accessible at [http://127.0.0.1:8000](http://127.0.0.1:8000)
* Interactive docs available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Calling the API

Below is a minimal example showing how to call the API once the service is running.

### Example using `curl`

```bash
curl -X POST "http://127.0.0.1:8000/process" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "doc_12345",
    "source_type": "pdf",
    "content": "<base64-encoded file or raw text>"
  }'
```

### Example using Python (`requests`)

```python
import requests

url = "http://127.0.0.1:8000/process"

payload = {
    "document_id": "doc_12345",
    "source_type": "pdf",
    "content": "<base64-encoded file or raw text>"
}

response = requests.post(url, json=payload)
response.raise_for_status()

print(response.json())
```

Refer to the OpenAPI documentation at `/docs` for the full request/response schema and available parameters.

## To-Do

* Further model training

## Current Issues

* Reconstruction is breaking text into paragraphs too aggressively
* (Fixed) extraction was causing character-level splitting (critical issue)
* requirements.txt is out of date
