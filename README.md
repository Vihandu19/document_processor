# document_processor
This project provides a end-to-end solution for transforming unstructured or messy documents (specifically PDFs and DOCX files) into clean, structured data formats (json and markdown) suitable for downstream applications like search indexing, data extraction, and analysis.

The core innovation is the use of a Machine Learning (ML) classifier to accurately determine the structural role of every line of text. Distinguish essential content (Body Text, Headings) from repetitive noise (ex. Headers, Footers, and Page Numbers), allowing for intelligent content cleanup

Sample Json ouput:
{
  "version": "1.0",
  "document_id": "doc_12345",

  "metadata": {
    "source_type": "pdf",
    "page_count": 12,
    "language": "en",
    "processed_at": "2025-01-01T12:00:00Z"
  },

  "sections": [
    {
      "id": "sec_1",
      "title": "Introduction",
      "level": 1,
      "page_start": 1,
      "page_end": 2,

      "content": [
        {
          "id": "blk_1",
          "type": "paragraph",
          "text": "... support@example.com. ...",
          "page": 1
        }
      ],

      "extraction_refs": {
        "emails": ["email_1"],
        "phone_numbers": [],
        "dates": [],
        "currency": [],
      }
    }
  ],

  "extractions": {
    "emails": [
      {
        "id": "email_1",
        "value": "support@example.com",
        "reliability": 0.98,
        "section_id": "sec_1",
        "block_id": "blk_1",
        "page": 1,
        "source": "regex"
      }
    ],

    "phone_numbers": [],
    "dates": [],
    "currency": [],
  },

  "removed": {
    "headers": [
      {
        "text": "Company Confidential",
        "pages": [...]
      }
    ],
    "footers": [
      {
        "text": "Page 1 of 12",
        "pages": [...]
      }
    ]
  }
}

Restrictions:
- Scanned/image-only PDFs are not supported

Prerequisites:
- Python 3.8+

Environment Activation
- source venv/bin/activate

Service Startup (API)
- uvicorn app.main:app --reload 
- The API will be accessible at http://127.0.0.1:8000. You can interact with the documentation interface at http://127.0.0.1:8000/docs.

to-do:
- further model training 

Current issues:
- reconstruction is breaking text into paragraphs to aggresivly 
- (fixed) extraction is causing character level splitting *critical issue*
- requirements.txt out of date 
