import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pharma_rag import PharmaGraphRAG
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    print("Debugging Pharmaceutical Graph Queries...")
    rag = PharmaGraphRAG()
    
    # Check what products are in the database
    products = rag.get_products()
    print(f"\nProducts in database: {products}")
    
    # Check what sales reps are in the database
    reps = rag.get_reps()
    print(f"\nSales reps in database: {reps}")
    
    # Check what HCPs are in the database
    hcps = rag.get_hcps()
    print(f"\nHCPs in database: {hcps}")
    
    # Test a direct product query
    if products:
        test_product = products[0]
        print(f"\nTesting direct query for product: {test_product}")
        
        cypher_query = """
            MATCH (p:Product {name: $product_name})
            MATCH (v:Visit)-[:DISCUSSED]->(p)
            MATCH (v)-[:RESULTED_IN]->(field:Insight {type: "Field"})
            MATCH (sr:SalesRep)-[:CONDUCTED]->(v)
            RETURN sr.name as rep_name, field.content as insight, v.date as date
            ORDER BY v.date DESC
        """
        
        results = rag.graph.query(cypher_query, {"product_name": test_product})
        print(f"Found {len(results)} insights")
        for result in results:
            print(f"- {result['rep_name']} reported: \"{result['insight']}\"")

if __name__ == "__main__":
    main() 