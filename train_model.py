import pandas as pd
import json
import joblib
import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import classification_report # <--- CRITICAL FIX: Missing import added
from app.processors.pdf_processor import extract_and_parse_pdf # <--- Moved to top for standard practice


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
    
    print(f"\nTraining with {len(final_feature_list)} features (after encoding)")
    
    #Splitting by index to avoid page-based data leakage (better for generalization)
    test_size = 0.2
    split_index = int(len(X) * (1 - test_size))

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    print(f"Training set: {len(X_train)} samples (first {split_index} lines)")
    print(f"Test set: {len(X_test)} samples (last {len(X) - split_index} lines, for better generalization)")
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=5,
        min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1
    )
    
    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
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
    
    # Save feature list and categorical metadata (CRITICAL FOR INFERENCE)
    ml_metadata = {
        'model_features': final_feature_list,
        'ohe_categories': {}
    }
    with open('model_metadata.json', 'w') as f:
        json.dump(ml_metadata, f, indent=2)
    print(f"✓ ML metadata saved to model_metadata.json (Total features: {len(final_feature_list)})")
    
    return model


def extract_features_from_pdf(pdf_path: str, output_json: str = "pdf_features.json"):
    """
    Extract features from a PDF and save to JSON for labeling.
    
    Args:
        pdf_path: Path to PDF file
        output_json: Where to save extracted features
    """
    # Removed local import since it's now at the top
    
    print(f"Extracting features from {pdf_path}...")
    
    with open(pdf_path, 'rb') as f:
        lines = extract_and_parse_pdf(f.read())
    
    print(f"Extracted {len(lines)} lines")
    
    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(lines, f, indent=2)
    
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

    #conver tot string for csv
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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train document structure classifier")
    parser.add_argument('--extract', type=str, help='Extract features from PDF')
    parser.add_argument('--auto-label', type=str, help='Auto-label JSON features and prepare CSV for manual review.')
    parser.add_argument('--prepare', type=str, help='Prepare CSV for labeling from JSON')
    parser.add_argument('--train', type=str, help='Train model from labeled CSV')
    
    args = parser.parse_args()
    
    if args.extract:
        # Step 1: Extract features from PDF
        json_path = extract_features_from_pdf(args.extract)
        print(f"\nNext: python train_model.py --prepare {json_path}")
    
    elif args.auto_label:
        # Step 2: Auto-label and Prepare CSV
        # This will call prepare_labeling_csv internally, saving label_me.csv
        auto_label_obvious(args.auto_label)
        print(f"\nNext: 1. Review and refine labels in label_me.csv")
        print(f"2. Run: python train_model.py --train label_me.csv")
        
    elif args.prepare:
        # Step 2: Prepare CSV for labeling
        prepare_labeling_csv(args.prepare)
        print(f"\nNext: python train_model.py --train label_me.csv")
        
    elif args.train:
        # Step 3: Train model
        train_model(args.train)
        print("\n✓ Training complete! Model is ready to use.")
        
    else:
        print("Usage:")
        print("  1. Extract: python train_model.py --extract document.pdf")
        print("  2. Auto-Label: python train_model.py --auto-label pdf_features.json")
        print("  3. Prepare: python train_model.py --prepare pdf_features.json")
        print("  4. Label manually in Excel/Sheets")
        print("  5. Train:   python train_model.py --train label_me.csv")