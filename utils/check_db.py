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
    print("Checking Pharmaceutical Graph Database...")
    rag = PharmaGraphRAG()
    
    # Check node counts
    node_counts = rag.graph.query("""
        MATCH (n)
        RETURN labels(n) as type, count(*) as count
    """)
    
    print("\nNode counts:")
    for record in node_counts:
        print(f"  {record['type']}: {record['count']}")
    
    # Check relationship counts
    rel_counts = rag.graph.query("""
        MATCH ()-[r]->()
        RETURN type(r) as type, count(*) as count
    """)
    
    print("\nRelationship counts:")
    for record in rel_counts:
        print(f"  {record['type']}: {record['count']}")
    
    # Check for specific relationships
    for rel_type in ["CONDUCTED", "WITH", "DISCUSSED", "GENERATED", "RESULTED_IN", "RELATES_TO"]:
        count = rag.graph.query(f"""
            MATCH ()-[r:{rel_type}]->()
            RETURN count(r) as count
        """)
        print(f"  {rel_type}: {count[0]['count'] if count else 0}")

    # Check for relationship types stored as properties
    rel_types = rag.graph.query("""
        MATCH ()-[r:RELATES_TO]->()
        RETURN r.relationship_type as type, count(*) as count
    """)
    
    print("\nRelationship types (as properties):")
    for record in rel_types:
        print(f"  {record['type']}: {record['count']}")

if __name__ == "__main__":
    main() 