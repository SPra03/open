import pandas as pd

def process_excel_to_csv(excel_file_path, output_csv_path):
    """
    Process an Excel file and convert it to a clean CSV file.
    
    Args:
        excel_file_path: Path to the Excel file
        output_csv_path: Path where the clean CSV will be saved
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Converting Excel file {excel_file_path} to CSV...")
    
    try:
        # Read the Excel file
        df = pd.read_excel(excel_file_path)
        
        # Check if we have the expected columns
        expected_columns = [
            "Sales Rep ID", 
            "Sales Rep Name", 
            "Product to Promote to Healthcare Provider",
            "XAI Insights Based on Model",
            "Field Insights Provided by Sales Rep"
        ]
        
        # Rename columns if they don't match exactly but have similar names
        for col in df.columns:
            for expected in expected_columns:
                if expected.lower() in col.lower():
                    df.rename(columns={col: expected}, inplace=True)
        
        # Check if all expected columns are present
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            print(f"ERROR: Missing columns in Excel file: {missing_columns}")
            return False
        
        # Save to CSV
        df.to_csv(output_csv_path, index=False)
        
        print(f"Successfully converted Excel to CSV. Saved to {output_csv_path}")
        print(f"Extracted {len(df)} data rows")
        return True
    
    except Exception as e:
        print(f"ERROR: Failed to process Excel file: {str(e)}")
        return False 