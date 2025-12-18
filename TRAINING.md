Internal Training pipeline (data ideally 1-50 pages and 200-1k lines)
1. python3 train_model.py --extract training_docs/training_data.pdf
1. or python3 train_model.py --predict-new training_docs/training_data.pdf (skip to step 3) 

2. python3 train_model.py --auto-label pdf_features.json 
2. or python3 train_model.py --prepare pdf_features.json
    
3. open label_me.csv (google sheets) label columns (0:regular content 1:header/footer 2:section title/heading)
5. save file to labeled_data/
4. repeat steps 1-3
5. python3 train_model.py --merge
6. python3 train_model.py --train master_labeled_data.csv

training data sources:
filetype:pdf "whitepaper"
filetype:pdf "user guide"
filetype:pdf "technical manual"
