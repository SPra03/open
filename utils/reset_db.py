from pharma_rag import PharmaGraphRAG
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    print("Completely resetting Neo4j database...")
    rag = PharmaGraphRAG()
    
    # Force delete everything
    rag.graph.query("""
        MATCH (n)
        DETACH DELETE n
    """)
    
    # Verify database is empty
    count = rag.graph.query("""
        MATCH (n)
        RETURN count(n) as count
    """)
    
    print(f"Database now has {count[0]['count']} nodes")
    
    if count[0]['count'] > 0:
        print("WARNING: Failed to completely clear database!")
    else:
        print("Database successfully reset")

if __name__ == "__main__":
    main() 