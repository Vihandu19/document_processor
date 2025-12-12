# train_model.py
import pandas as pd
import json
import joblib
import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


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
    Train Random Forest classifier from labeled CSV.
    
    Args:
        labeled_csv_path: Path to CSV with 'label' column filled in
        model_output: Where to save the trained model
    """
    # Load labeled data
    df = pd.read_csv(labeled_csv_path)
    
    # Check if labels exist
    if 'label' not in df.columns:
        raise ValueError(f"'{labeled_csv_path}' must have a 'label' column!")
    
    # Remove unlabeled rows
    df = df[df['label'].notna() & (df['label'] != '')]
    df['label'] = df['label'].astype(int)
    
    print(f"Loaded {len(df)} labeled lines")
    print(f"Label distribution:\n{df['label'].value_counts().sort_index()}")
    
    # Check if we have all classes
    if len(df['label'].unique()) < 3:
        print("\n⚠ WARNING: Not all classes (0,1,2) are present in training data!")
        print("This might affect model performance.")
    
    # Select features for training
    feature_columns = [
        # Layout
        'y_pos_norm', 'x_pos_norm', 'line_width_norm', 'is_centered',
        
        # Text
        'char_count', 'word_count', 'is_short', 'is_very_short', 'is_single_word',
        
        # Style
        'font_size', 'is_bold', 'is_italic',
        
        # Content
        'is_all_caps', 'is_title_case', 'is_numeric_pattern',
        'has_page_keyword', 'has_date_pattern', 'starts_with_number', 'ends_with_colon',
        
        # Position
        'is_top_10_percent', 'is_bottom_10_percent', 'is_first_page', 'is_last_page',
        
        # Frequency
        'signature_frequency', 'appears_on_multiple_pages', 'appears_on_most_pages'
    ]
    
    # Only use features that exist in the dataframe
    available_features = [f for f in feature_columns if f in df.columns]
    missing_features = [f for f in feature_columns if f not in df.columns]
    
    if missing_features:
        print(f"\n⚠ Missing features: {missing_features}")
        print("Make sure your PDF extraction includes all features!")
    
    X = df[available_features]
    y = df['label']
    
    print(f"\nTraining with {len(available_features)} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',  # Handle imbalanced classes
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    
    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(
        y_test, y_pred, 
        target_names=['Content', 'Header/Footer', 'Title'],
        digits=3
    ))
    
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    cm = confusion_matrix(y_test, y_pred)
    print("          Predicted")
    print("         Content  H/F  Title")
    for i, row in enumerate(cm):
        label_name = ['Content', 'H/F', 'Title'][i]
        print(f"Actual {label_name:8s} {row[0]:4d}   {row[1]:4d}  {row[2]:4d}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n" + "="*60)
    print("TOP 15 MOST IMPORTANT FEATURES")
    print("="*60)
    for idx, row in feature_importance.head(15).iterrows():
        print(f"{row['feature']:30s} {row['importance']:.4f}")
    
    # Save model
    joblib.dump(model, model_output)
    print(f"\n✓ Model saved to {model_output}")
    
    # Save feature list (needed for inference)
    with open('model_features.json', 'w') as f:
        json.dump(available_features, f)
    print(f"✓ Feature list saved to model_features.json")
    
    return model


def extract_features_from_pdf(pdf_path: str, output_json: str = "pdf_features.json"):
    """
    Extract features from a PDF and save to JSON for labeling.
    
    Args:
        pdf_path: Path to PDF file
        output_json: Where to save extracted features
    """
    from app.processors.pdf_processor import extract_and_parse_pdf
    
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

    # ---- CONTENT (label = 0) ----
    df.loc[df['label'] == -1, 'label'] = 0

    # Convert to strings for spreadsheet editing
    df['label'] = df['label'].astype(str)

    # ---- SAVE back to NEW JSON ----
    autolabeled_json = features_json_path.replace(".json", "_autolabeled.json")

    with open(autolabeled_json, 'w') as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)

    print(f"✓ Auto-labeled JSON written to {autolabeled_json}")

    # ---- Now generate CSV using existing function ----
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
    parser.add_argument('--prepare', type=str, help='Prepare CSV for labeling from JSON')
    parser.add_argument('--train', type=str, help='Train model from labeled CSV')
    
    args = parser.parse_args()
    
    if args.extract:
        # Step 1: Extract features from PDF
        json_path = extract_features_from_pdf(args.extract)
        print(f"\nNext: python train_model.py --prepare {json_path}")
        
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
        print("  2. Prepare: python train_model.py --prepare pdf_features.json")
        print("  3. Label manually in Excel/Sheets")
        print("  4. Train:   python train_model.py --train label_me.csv")