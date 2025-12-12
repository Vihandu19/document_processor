# document_processor
This project provides a end-to-end solution for transforming unstructured or messy documents (specifically PDFs and DOCX files) into clean, structured data formats (json and markdown) suitable for downstream applications like search indexing, data extraction, and analysis.

The core innovation is the use of a Machine Learning (ML) classifier to accurately determine the structural role of every line of text. Distinguish essential content (Body Text, Headings) from unwanted, repetitive noise (ex.Headers, Footers, and Page Numbers), allowing for intelligent content cleanup that goes far beyond rules based parsing.

Prerequisites:
    Python 3.8+
    The required libraries must be installed ( pip install -r requirements.txt).

Environment Activation
    source venv/bin/activate

Service Startup (API)
    uvicorn app.main:app --reload 
    The API will be accessible at http://127.0.0.1:8000. You can interact with the documentation interface at http://127.0.0.1:8000/docs.

model train (one file at a time due to lack of line shuffling)
    python train_model.py --extract document.pdf
    python train_model.py --auto-label pdf_features.json
    python train_model.py --prepare pdf_features.json
    
    open label_me.csv (google sheets) label columns
    0:regular content
    1:header/footer
    2:section title/heading
    save file as label_me.csv
    
    python train_model.py --train label_me.csv

to-do:
- train model for header/footer removal and section idenfitication
- modify cleantext to use ml instead of rules based 
- update cleanup.py functions to take json as input 

Current issues:
- requirements.txt out of date 

