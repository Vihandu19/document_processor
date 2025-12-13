import pandas as pd
import glob
import os

def merge_csvs(input_directory, output_filename="master_labeled_data.csv"):
    """
    Finds all CSV files in a directory, merges them into a single DataFrame,
    and saves the result to a master CSV file.
    """
    all_files = glob.glob(os.path.join(input_directory, "*.csv"))

    if not all_files:
        print(f"❌ Error: No CSV files found in the directory: {input_directory}")
        return

    # List to hold individual DataFrames
    all_dataframes = []

    # Read each CSV file
    for filename in all_files:
        try:
            # We assume all your labeled CSVs have the same structure
            df = pd.read_csv(filename)
            
            # Important: Filter out any lines that are still unlabeled or blank
            df = df[df['label'].astype(str).str.fullmatch(r'[012]', na=False)]
            df['label'] = df['label'].astype(int)
            
            all_dataframes.append(df)
            print(f"Loaded {len(df)} labeled lines from: {os.path.basename(filename)}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not read {filename}. Skipping. Error: {e}")

    # Concatenate all DataFrames into one
    master_df = pd.concat(all_dataframes, ignore_index=True)

    # Save the master file
    master_df.to_csv(output_filename, index=False)
    
    # Report totals
    print("-" * 40)
    print(f"✅ Merging Complete.")
    print(f"Total labeled lines: {len(master_df)}")
    print(f"Label distribution:\n{master_df['label'].value_counts().sort_index()}")
    print(f"Saved to: {output_filename}")


if __name__ == "__main__":
    merge_csvs('labeled_data')