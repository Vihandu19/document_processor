# document_processor
To run:
source venv/bin/activate
uvicorn app.main:app --reload 

to train(one file at a time due to lack of line shuffling):
python train_model.py --extract document.pdf
python train_model.py --auto-label pdf_features.json
python train_model.py --prepare pdf_features.json
***
open label_me.csv (google sheets) label columns
0:regular content
1: header/footer
2:section title/heading
save file as label_me.csv
***
python train_model.py --train label_me.csv


to-do:
train model for header/footer removal and section idenfitication

Current issues:

