import pandas as pd
import json
import joblib
import argparse
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import classification_report 
from app.processors.pdf_processor import extract_and_parse_pdf,extract_and_parse_pdf_linebyline
import glob
import os


def merge_labeled_csvs(input_directory: str = "labeled_data", output_filename: str = "master_labeled_data.csv"):
    """
    Finds all CSV files in a directory, merges them into a single DataFrame,
    and saves the result to a master CSV file.
    IMPORTANT: Adds 'document_id' and 'source_file' columns to track document origin.
    """
    all_files = sorted(glob.glob(os.path.join(input_directory, "*.csv")))

    if not all_files:
        print(f"❌ Error: No CSV files found in the directory: {input_directory}")
        return None

    # List to hold individual DataFrames
    all_dataframes = []

    # Read each CSV file with document tracking
    for doc_id, filename in enumerate(all_files, start=1):
        try:
            # We assume all your labeled CSVs have the same structure
            df = pd.read_csv(filename)
            
            # Important: Filter out any lines that are still unlabeled or blank
            original_count = len(df)
            df = df[df['label'].astype(str).str.fullmatch(r'[012]', na=False)]
            df['label'] = df['label'].astype(int)
            filtered_count = len(df)
            
            # Add document tracking columns
            df['document_id'] = doc_id
            df['source_file'] = os.path.basename(filename)
            
            all_dataframes.append(df)
            
            print(f"Doc {doc_id}: Loaded {filtered_count} labeled lines from: {os.path.basename(filename)}")
            if original_count != filtered_count:
                print(f"        ⚠️  Filtered out {original_count - filtered_count} unlabeled/invalid rows")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not read {filename}. Skipping. Error: {e}")

    if not all_dataframes:
        print(f"❌ Error: No valid data to merge!")
        return None

    # Concatenate all DataFrames into one
    master_df = pd.concat(all_dataframes, ignore_index=True)

    # Save the master file
    master_df.to_csv(output_filename, index=False)
    
    # Report totals
    print("-" * 60)
    print(f"✅ Merging Complete.")
    print(f"Total labeled lines: {len(master_df)}")
    print(f"Total documents: {master_df['document_id'].nunique()}")
    print(f"\nLabel distribution:")
    print(master_df['label'].value_counts().sort_index())
    print(f"\nDocument distribution:")
    print(master_df['document_id'].value_counts().sort_index())
    print(f"\nSaved to: {output_filename}")
    
    return output_filename


def prepare_labeling_csv(features_json_path: str, output_csv: str = "label_me.csv"):
    """
    Convert extracted PDF features to CSV for manual labeling.
    Keeps all features but shows important ones first for easy labeling.
    """
    # Load features
    with open(features_json_path, 'r') as f:
        lines = json.load(f)
    
    df = pd.DataFrame(lines)
    
    # Add empty label column if it doesn't exist
    if 'label' not in df.columns:
        df['label'] = ''
    
    # Priority columns for viewing (these go first for easy labeling)
    priority_columns = [
        'label',  # Put this FIRST so it's easy to fill in
        'page_num',
        'y_pos_norm',
        'text_content',
        'char_count',
        'font_size',
        'is_all_caps',
        'is_centered',
        'is_top_10_percent',
        'is_bottom_10_percent',
        'signature_frequency',
    ]
    
    # Get all other columns
    other_columns = [col for col in df.columns if col not in priority_columns]
    
    # Reorder: priority columns first, then everything else
    ordered_columns = [col for col in priority_columns if col in df.columns] + other_columns
    df_reordered = df[ordered_columns]
    
    # Save with ALL features (needed for training)
    df_reordered.to_csv(output_csv, index=False)
    
    print(f"✓ Saved {len(df_reordered)} lines to {output_csv}")
    print(f"  Total columns: {len(df_reordered.columns)}")
    print(f"\nNext steps:")
    print(f"1. Open {output_csv} in Excel/Google Sheets")
    print(f"2. Fill in the 'label' column (first column):")
    print(f"   - 0 = Regular content (paragraph text)")
    print(f"   - 1 = Header/Footer (should be removed)")
    print(f"   - 2 = Section title/heading")
    print(f"3. You can ignore all other columns - just label!")
    print(f"4. Save the file (keep same name: {output_csv})")
    print(f"5. Run: python train_model.py --train {output_csv}")
    
    return df_reordered


def train_model(labeled_csv_path: str, model_output: str = "document_structure_model.pkl"):
    """
    Train a robust Random Forest classifier from labeled CSV, correctly handling
    categorical features and ensuring a stable feature set for inference.
    
    Args:
        labeled_csv_path: Path to CSV with 'label' column filled in
        model_output: Where to save the trained model
    """
    # Load labeled data
    df = pd.read_csv(labeled_csv_path)
    
    # Validation and Label Cleaning
    if 'label' not in df.columns:
        raise ValueError(f"'{labeled_csv_path}' must have a 'label' column!")
    
    df = df[df['label'].notna() & (df['label'] != '')]
    df['label'] = df['label'].astype(int)
    
    # CRITICAL FIX: Check if there is any labeled data left
    if len(df) == 0:
        raise ValueError("No labeled data found. Please ensure the CSV has integer values in the 'label' column and retry.")

    print(f"Loaded {len(df)} labeled lines")
    print(f"Label distribution:\n{df['label'].value_counts().sort_index()}")
    
    
    #Define the baseline set of features (all numerical/boolean)
    BASE_NUMERIC_FEATURES = [
        'y_pos_norm', 'x_pos_norm', 'line_width_norm', 'is_centered', 'is_top_10_percent', 
        'is_bottom_10_percent', 'is_first_page', 'is_last_page',
        'char_count', 'word_count', 'is_short', 'is_very_short', 'is_single_word',
        'font_size', 'is_bold', 'is_italic',
        'is_all_caps', 'is_title_case', 'is_numeric_pattern',
        'has_page_keyword', 'has_date_pattern', 'starts_with_number', 'ends_with_colon',
        'signature_frequency', 'appears_on_multiple_pages', 'appears_on_most_pages'
    ]
    
    # Features to drop entirely
    DROP_FEATURES = ['text_content', 'line_signature'] 
    
    
    # Drop high-cardinality/redundant/text features
    df_processed = df.drop(columns=DROP_FEATURES, errors='ignore')
    
    # Get available features that exist in the loaded data
    available_base_features = [f for f in BASE_NUMERIC_FEATURES if f in df_processed.columns]
    
    final_feature_list = available_base_features

    # Ensure all final features exist (crucial for inference)
    for col in final_feature_list:
        if col not in df_processed.columns:
            df_processed[col] = 0

    # Build final X, y matrices (fillna(0) handles potential NaN from merged features)
    X = df_processed[final_feature_list].fillna(0)
    y = df_processed['label']
    
    print(f"\nTraining with {len(final_feature_list)} features")
    
    # CRITICAL: Check if we have document_id column for proper splitting
    if 'document_id' in df_processed.columns:
        print("\n✓ Found 'document_id' column - using document-level split (RECOMMENDED)")
        
        # Get unique document IDs
        unique_docs = df_processed['document_id'].unique()
        n_docs = len(unique_docs)
        
        # Split documents 80/20
        test_size = 0.2
        n_test_docs = max(1, int(n_docs * test_size))
        n_train_docs = n_docs - n_test_docs
        
        # Randomly shuffle documents
        np.random.seed(42)
        shuffled_docs = np.random.permutation(unique_docs)
        
        train_docs = shuffled_docs[:n_train_docs]
        test_docs = shuffled_docs[n_train_docs:]
        
        # Split data by document
        train_mask = df_processed['document_id'].isin(train_docs)
        test_mask = df_processed['document_id'].isin(test_docs)
        
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        print(f"Document-level split:")
        print(f"  Training: {n_train_docs} documents ({len(X_train)} lines)")
        print(f"  Test:     {n_test_docs} documents ({len(X_test)} lines)")
        print(f"  This prevents data leakage from same-document lines!")
        
    else:
        print("\n⚠️  WARNING: No 'document_id' column found!")
        print("   Using simple index-based split (may cause data leakage)")
        print("   RECOMMENDATION: Re-merge your data with document_id tracking")
        
        # Fallback to index-based splitting
        test_size = 0.2
        split_index = int(len(X) * (1 - test_size))
        
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        
        print(f"Training set: {len(X_train)} samples (first {split_index} lines)")
        print(f"Test set: {len(X_test)} samples (last {len(X) - split_index} lines)")
    
    # Train model (OUTSIDE the if/else blocks - runs regardless of split method)
    model = RandomForestClassifier(
        n_estimators=50,      # Conservative for small datasets
        max_depth=8,          # Shallower trees
        min_samples_split=10, # Need more samples to split
        min_samples_leaf=5,   # Bigger leaves
        class_weight='balanced', 
        random_state=42, 
        n_jobs=-1
    )
    
    # Cross-validation before training (helps assess model stability)
    print("\nRunning 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted', n_jobs=-1)
    print(f"CV F1 Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"Individual folds: {[f'{score:.3f}' for score in cv_scores]}")
    
    if cv_scores.std() > 0.10:
        print("⚠️  High variance detected - model may be unstable")
    if cv_scores.mean() < 0.70:
        print("⚠️  Low F1 score - consider adding more labeled data or reviewing features")
    
    print("\nTraining Random Forest on full training set...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE")
    print("="*60)
    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Test accuracy:     {test_acc:.3f}")
    print(f"Accuracy gap:      {train_acc - test_acc:.3f}")
    
    if train_acc - test_acc > 0.15:
        print("⚠️  Large gap detected - model may be overfitting")
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(
        y_test, y_pred, 
        target_names=['Content', 'Header/Footer', 'Title'],
        digits=3
    ))
    
    # Save model
    joblib.dump(model, model_output)
    print(f"\n✓ Model saved to {model_output}")
    
    # Save feature list (CRITICAL FOR INFERENCE)
    ml_metadata = {
        'model_features': final_feature_list
    }
    with open('model_metadata.json', 'w') as f:
        json.dump(ml_metadata, f, indent=2)
    print(f"✓ ML metadata saved to model_metadata.json (Total features: {len(final_feature_list)})")
    
    return model


def extract_features_from_pdf(pdf_path: str, output_json: str = "pdf_features.json"):
    print(f"Extracting features from {pdf_path}...")

    # 1. Read PDF bytes
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()

    # 2. Extract features (ONLY here)
    lines = extract_and_parse_pdf_linebyline(
        pdf_bytes,
        x_tolerance=8,
        y_tolerance=5
    )

    print(f"Extracted {len(lines)} lines")

    # 3. Write JSON safely
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(lines, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved features to {output_json}")

    return output_json



def auto_label_obvious(features_json_path: str, output_csv: str = "label_me.csv"):
    """
    Auto-label obvious document elements and write a new JSON file before CSV prep.
    Works with the existing prepare_labeling_csv() which expects a JSON path.
    """
    # Load original JSON
    with open(features_json_path, 'r') as f:
        lines = json.load(f)

    df = pd.DataFrame(lines)

    # Safe access helper so missing features don't crash
    def col(name, default=False):
        return df[name] if name in df.columns else default

    df['label'] = -1  # temporary placeholder

    # ---- HEADER / FOOTER (label = 1) ----
    header_footer_mask = (
        col('appears_on_most_pages') &
        (col('is_top_10_percent') | col('is_bottom_10_percent')) &
        (col('char_count') < 50)
    ) | col('is_numeric_pattern')

    df.loc[header_footer_mask, 'label'] = 1

    # ---- TITLE (label = 2) ----
    title_mask = (
        (df['label'] == -1) &          # don't override header/footer
        col('is_centered') &
        col('is_all_caps') &
        col('char_count').between(5, 80)
    )

    df.loc[title_mask, 'label'] = 2

    # ---- REGULAR CONTENT (label = 0) ----
    df.loc[df['label'] == -1, 'label'] = 0

    #convert to string for csv
    df['label'] = df['label'].astype(str)

    #save back to json
    autolabeled_json = features_json_path.replace(".json", "_autolabeled.json")

    with open(autolabeled_json, 'w') as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)

    print(f"✓ Auto-labeled JSON written to {autolabeled_json}")

    #generate CSV
    prepare_labeling_csv(autolabeled_json, output_csv)

    print("\n📊 Auto-labeling summary:")
    print(f"  Header/Footer (1): {(df['label'] == '1').sum()}")
    print(f"  Titles (2):        {(df['label'] == '2').sum()}")
    print(f"  Content (0):       {(df['label'] == '0').sum()}")

    print("\n⚠️  Review labels in the CSV — these are only heuristics.")


def auto_label_with_model(pdf_path: str, model_path: str = "document_structure_model.pkl", metadata_path: str = "model_metadata.json", output_csv: str = "label_me_predicted.csv"):
    """
    Uses an EXISTING model to auto-label a NEW PDF.
    Saves the result as a CSV for human review/correction.
    """
    import os
    
    # 1. Check if model exists
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        print(f"❌ Model or metadata not found. Train a model first")
        return

    # 2. Extract features from the new PDF
    print(f"Extracting features from {pdf_path}...")
    
    with open(pdf_path, 'rb') as f:
        raw_lines = extract_and_parse_pdf_linebyline(f.read())
    
    df = pd.DataFrame(raw_lines)
    print(f"Extracted {len(df)} lines.")

    # 3. Load Model and Metadata
    print("Loading model and metadata...")
    model = joblib.load(model_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    required_features = metadata['model_features']

    # 4. Preprocess Features (MUST MATCH TRAINING EXACTLY)
    
    # Drop non-numeric/high-cardinality cols (same as training)
    DROP_FEATURES = ['text_content', 'line_signature']
    inference_data = df.drop(columns=DROP_FEATURES, errors='ignore')
    
    # Align columns: ensure we have exactly the features the model expects
    # Missing features are filled with 0
    for feature in required_features:
        if feature not in inference_data.columns:
            inference_data[feature] = 0
    
    # Select only the required features in the correct order
    X = inference_data[required_features].fillna(0)
    
    # 5. Predict
    print("Predicting labels...")
    predicted_labels = model.predict(X)
    
    # 6. Save results to CSV for review
    # We put the predicted label in the first column
    df['label'] = predicted_labels
    
    # Reorder for readability (same logic as prepare_labeling_csv)
    priority_columns = [
        'label', 'page_num', 'y_pos_norm', 'text_content', 
        'font_size', 'is_bold', 'is_all_caps'
    ]
    other_columns = [c for c in df.columns if c not in priority_columns]
    ordered_columns = [col for col in priority_columns if col in df.columns] + other_columns
    
    final_df = df[ordered_columns]
    
    final_df.to_csv(output_csv, index=False)
    
    print(f"\n✓ Generated pre-labeled file: {output_csv}")
    print(f"  Label distribution:")
    print(f"    Content (0):       {(predicted_labels == 0).sum()}")
    print(f"    Header/Footer (1): {(predicted_labels == 1).sum()}")
    print(f"    Title (2):         {(predicted_labels == 2).sum()}")
    print(f"\n  Next steps:")
    print(f"  1. Open {output_csv}")
    print(f"  2. Review the 'label' column. Correct any mistakes.")
    print(f"     (0=Content, 1=Header/Footer, 2=Title)")
    print(f"  3. Save and use for re-training if needed!")




if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train document structure classifier")
    parser.add_argument('--extract', type=str, help='Extract features from PDF')
    parser.add_argument('--auto-label', type=str, help='Auto-label JSON features and prepare CSV for manual review.')
    parser.add_argument('--predict-new', type=str, help='Use TRAINED MODEL to label a new PDF')
    parser.add_argument('--prepare', type=str, help='Prepare CSV for labeling from JSON')
    parser.add_argument('--merge', type=str, nargs='?', const='labeled_data', help='Merge labeled CSVs from directory (default: labeled_data)')
    parser.add_argument('--train', type=str, help='Train model from labeled CSV')
    
    args = parser.parse_args()
    
    if args.extract:
        # Extract features from PDF
        json_path = extract_features_from_pdf(args.extract)
    
    elif args.auto_label:
        # Auto-label and Prepare CSV
        # This will call prepare_labeling_csv internally, saving label_me.csv
        auto_label_obvious(args.auto_label)
    
    elif args.predict_new:
        auto_label_with_model(args.predict_new)
    
    elif args.merge is not None:
        # Merge labeled CSVs
        output_file = merge_labeled_csvs(args.merge)
        
    elif args.prepare:
        # Step 2: Prepare CSV for labeling
        prepare_labeling_csv(args.prepare)
        
    elif args.train:
        # Step 3: Train model
        train_model(args.train)
        
    else:
        print("Usage:")
        print("  1. Extract:   python train_model.py --extract document.pdf")
        print("  2. Auto-Label: python train_model.py --auto-label pdf_features.json")
        print("  3. Prepare:   python train_model.py --prepare pdf_features.json")
        print("  4. Label manually in Excel/Sheets")
        print("  5. Merge:     python train_model.py --merge [directory]")
        print("  6. Train:     python train_model.py --train label_me.csv")