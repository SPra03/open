from pharma_rag import PharmaGraphRAG
import os
from dotenv import load_dotenv
from rtf_parser import parse_rtf_to_csv
from excel_parser import process_excel_to_csv

# Load environment variables
load_dotenv()

def main():
    print("Setting up Pharmaceutical Graph Database...")
    rag = PharmaGraphRAG()
    
    # Clear existing data
    rag.clear_database()
    print("Database cleared")
    
    # Check for data files in different formats
    pharma_excel = "pharma_data.xlsx"
    pharma_rtf = "pharma_data.csv"  # The RTF file with .csv extension
    clean_csv = "clean_pharma_data.csv"
    
    # Try Excel first
    if os.path.exists(pharma_excel):
        # Convert Excel to clean CSV
        if process_excel_to_csv(pharma_excel, clean_csv):
            print(f"Processing {clean_csv}...")
            rag.process_pharma_data(clean_csv)
            print("Data processing complete")
    # Fall back to RTF if Excel doesn't exist
    elif os.path.exists(pharma_rtf):
        # First convert RTF to clean CSV
        if parse_rtf_to_csv(pharma_rtf, clean_csv):
            print(f"Processing {clean_csv}...")
            rag.process_pharma_data(clean_csv)
            print("Data processing complete")
    else:
        print(f"Error: No data file found. Please provide either {pharma_excel} or {pharma_rtf}")
        return
    
    # Verify data was loaded
    node_count = rag.graph.query("""
        MATCH (n) RETURN count(n) as count
    """)
    rel_count = rag.graph.query("""
        MATCH ()-[r]->() RETURN count(r) as count
    """)
    
    print(f"Loaded {node_count[0]['count']} nodes and {rel_count[0]['count']} relationships")
    
    if node_count[0]['count'] == 0 or rel_count[0]['count'] == 0:
        print("WARNING: Database appears to be empty after loading!")

if __name__ == "__main__":
    main() 