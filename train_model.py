import pandas as pd
import json
import joblib
import argparse
import numpy as np
import glob
import os
import pdfplumber
from huggingface_hub import snapshot_download
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report 
from app.processors.pdf_processor import extract_and_parse_pdf, extract_and_parse_pdf_linebyline


def merge_labeled_csvs(input_directory: str = "labeled_data", output_filename: str = "master_labeled_data.csv"):
    all_files = sorted(glob.glob(os.path.join(input_directory, "*.csv")))
    if not all_files:
        print(f"❌ Error: No CSV files found in: {input_directory}")
        return None

    all_dataframes = []
    for doc_id, filename in enumerate(all_files, start=1):
        try:
            df = pd.read_csv(filename)
            df = df[df['label'].astype(str).str.fullmatch(r'[012]', na=False)]
            df['label'] = df['label'].astype(int)
            df['document_id'] = doc_id
            df['source_file'] = os.path.basename(filename)
            all_dataframes.append(df)
        except Exception as e:
            print(f"⚠️ Warning: Skipping {filename}. Error: {e}")

    master_df = pd.concat(all_dataframes, ignore_index=True)
    master_df.to_csv(output_filename, index=False)
    print(f"✅ Merged {len(master_df)} lines from {master_df['document_id'].nunique()} docs.")
    return output_filename

def train_model(labeled_csv_path: str, model_output: str = "document_structure_model.pkl"):
    df = pd.read_csv(labeled_csv_path)
    df = df[df['label'].notna() & (df['label'] != '')]
    df['label'] = df['label'].astype(int)

    BASE_NUMERIC_FEATURES = [
        'y_pos_norm', 'x_pos_norm', 'line_width_norm', 'is_centered', 'is_top_10_percent', 
        'is_bottom_10_percent', 'is_first_page', 'is_last_page',
        'char_count', 'word_count', 'is_short', 'is_very_short', 'is_single_word',
        'font_size', 'is_bold', 'is_italic',
        'is_all_caps', 'is_title_case', 'is_numeric_pattern',
        'has_page_keyword', 'has_date_pattern', 'starts_with_number', 'ends_with_colon',
        'signature_frequency', 'appears_on_multiple_pages', 'appears_on_most_pages',
        'font_size_rel', 'prev_ends_punctuation', 'next_starts_lower', 'dist_from_prev', 
        'font_size_change_prev', 'is_different_style_prev' 
    ]
    
    available_features = [f for f in BASE_NUMERIC_FEATURES if f in df.columns]
    X = df[available_features].fillna(0)
    y = df['label']
    
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, model_output)
    with open('model_metadata.json', 'w') as f:
        json.dump({'model_features': available_features}, f, indent=2)
    print(f"✓ Model and Metadata saved.")

def extract_features_from_pdf(pdf_path: str, output_json: str = "pdf_features.json"):
    with open(pdf_path, 'rb') as f:
        lines = extract_and_parse_pdf_linebyline(f.read())
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(lines, f, indent=2, ensure_ascii=False)
    return output_json

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Intelligence Training CLI")
   
    parser.add_argument('--extract', type=str, help='Extract features from PDF')
    parser.add_argument('--merge', type=str, nargs='?', const='labeled_data', help='Merge labeled CSVs')
    parser.add_argument('--train', type=str, help='Train model from labeled CSV')
    
    args = parser.parse_args()
    
    if args.download_data:
        download_cuad_data()

    if args.clean:
        clean_dataset(args.clean, args.max_pages)

    if args.extract:
        extract_features_from_pdf(args.extract)
    
    if args.merge:
        merge_labeled_csvs(args.merge)
        
    if args.train:
        train_model(args.train)