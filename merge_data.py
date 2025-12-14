import pandas as pd
import glob
import os

def merge_csvs(input_directory, output_filename="master_labeled_data.csv"):
    """
    Finds all CSV files in a directory, merges them into a single DataFrame,
    and saves the result to a master CSV file.
    IMPORTANT: Adds 'document_id' and 'source_file' columns to track document origin.
    """
    all_files = glob.glob(os.path.join(input_directory, "*.csv"))

    if not all_files:
        print(f"❌ Error: No CSV files found in the directory: {input_directory}")
        return

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
            
            #Add document tracking columns
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
        return

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


if __name__ == "__main__":
    merge_csvs('labeled_data')