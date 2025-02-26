import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pharma_rag import PharmaGraphRAG
import os
from dotenv import load_dotenv
from excel_parser import process_excel_to_csv

# Load environment variables
load_dotenv()

def main():
    print("=== COMPLETE DATABASE RESET AND VERIFICATION ===")
    rag = PharmaGraphRAG()
    
    # Step 1: Reset the database
    print("\n1. Resetting database...")
    rag.graph.query("""
        MATCH (n)
        DETACH DELETE n
    """)
    
    # Verify database is empty
    count = rag.graph.query("""
        MATCH (n)
        RETURN count(n) as count
    """)
    
    if count[0]['count'] > 0:
        print(f"ERROR: Failed to clear database. Still has {count[0]['count']} nodes.")
        return
    else:
        print("✓ Database successfully cleared")
    
    # Step 2: Process data files
    print("\n2. Processing data files...")
    
    # Check for data files in different formats
    pharma_excel = "pharma_data.xlsx"
    pharma_rtf = "pharma_data.csv"
    clean_csv = "clean_pharma_data.csv"
    
    data_processed = False
    
    # Try Excel first
    if os.path.exists(pharma_excel):
        print(f"Found Excel file: {pharma_excel}")
        if process_excel_to_csv(pharma_excel, clean_csv):
            print(f"Processing {clean_csv}...")
            rag.process_pharma_data(clean_csv)
            data_processed = True
    
    # Fall back to RTF if Excel doesn't exist or failed
    if not data_processed and os.path.exists(pharma_rtf):
        print(f"Found RTF file: {pharma_rtf}")
        if parse_rtf_to_csv(pharma_rtf, clean_csv):
            print(f"Processing {clean_csv}...")
            rag.process_pharma_data(clean_csv)
            data_processed = True
    
    if not data_processed:
        print("ERROR: No data files processed. Please provide either pharma_data.xlsx or pharma_data.csv")
        return
    
    # Step 3: Verify data was loaded correctly
    print("\n3. Verifying data...")
    
    # Check nodes
    node_counts = rag.graph.query("""
        MATCH (n)
        RETURN labels(n)[0] as type, count(*) as count
    """)
    
    print("Node counts:")
    for record in node_counts:
        print(f"  {record['type']}: {record['count']}")
    
    # Check relationships
    rel_counts = rag.graph.query("""
        MATCH ()-[r]->()
        RETURN type(r) as type, count(*) as count
    """)
    
    print("\nRelationship counts:")
    for record in rel_counts:
        print(f"  {record['type']}: {record['count']}")
    
    # Step 4: Test a product query
    print("\n4. Testing product query...")
    
    products = rag.get_products()
    if not products:
        print("ERROR: No products found in database")
        return
    
    test_product = products[0]
    print(f"Testing query for product: {test_product}")
    
    cypher_query = """
        MATCH (p:Product {name: $product_name})
        MATCH (v:Visit)-[:DISCUSSED]->(p)
        MATCH (v)-[:RESULTED_IN]->(field:Insight {type: "Field"})
        MATCH (sr:SalesRep)-[:CONDUCTED]->(v)
        RETURN sr.name as rep_name, field.content as insight
        LIMIT 1
    """
    
    results = rag.graph.query(cypher_query, {"product_name": test_product})
    if results:
        print(f"✓ Successfully queried insights for {test_product}")
        print(f"Sample insight: {results[0]['insight'][:100]}...")
    else:
        print(f"ERROR: No insights found for {test_product}")
        
        # Try a simpler query to diagnose the issue
        print("\nDiagnosing relationship issues...")
        
        # Check if product exists
        product_check = rag.graph.query("""
            MATCH (p:Product {name: $product_name})
            RETURN p
        """, {"product_name": test_product})
        
        if not product_check:
            print(f"ERROR: Product '{test_product}' not found in database")
        else:
            print(f"✓ Product '{test_product}' exists in database")
        
        # Check DISCUSSED relationship
        discussed_check = rag.graph.query("""
            MATCH (p:Product {name: $product_name})
            MATCH (v:Visit)-[:DISCUSSED]->(p)
            RETURN count(v) as count
        """, {"product_name": test_product})
        
        if discussed_check[0]['count'] == 0:
            print(f"ERROR: No DISCUSSED relationships found for {test_product}")
        else:
            print(f"✓ Found {discussed_check[0]['count']} DISCUSSED relationships")
        
        # Check RESULTED_IN relationship
        resulted_check = rag.graph.query("""
            MATCH (v:Visit)
            MATCH (v)-[:RESULTED_IN]->(field:Insight {type: "Field"})
            RETURN count(field) as count
        """)
        
        if resulted_check[0]['count'] == 0:
            print(f"ERROR: No RESULTED_IN relationships found")
        else:
            print(f"✓ Found {resulted_check[0]['count']} RESULTED_IN relationships")
    
    print("\n=== VERIFICATION COMPLETE ===")

if __name__ == "__main__":
    main() 