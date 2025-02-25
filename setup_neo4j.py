from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

def setup_neo4j_schema():
    print("Setting up Neo4j schema...")
    try:
        # Test the connection parameters
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        
        if not all([uri, username, password]):
            raise ValueError("Missing required environment variables. Please check your .env file.")
        
        print("Connecting with parameters:")
        print(f"URI: {uri}")
        print(f"Username: {username}")
        print("Password: ********")
        
        # Try to establish connection
        driver = GraphDatabase.driver(
            uri,
            auth=(username, password)
        )
        
        # Verify connection
        print("Testing connection...")
        driver.verify_connectivity()
        print("Connection successful!")
        
        with driver.session() as session:
            # Create constraints
            print("Creating constraints...")
            session.run("""
                CREATE CONSTRAINT topic_name IF NOT EXISTS
                FOR (t:Topic) REQUIRE t.name IS UNIQUE
            """)
            
            # Create indexes
            print("Creating indexes...")
            session.run("""
                CREATE INDEX topic_name_idx IF NOT EXISTS
                FOR (t:Topic) ON (t.name)
            """)
        
        driver.close()
        print("Neo4j schema setup completed successfully!")
        
    except Exception as e:
        print(f"\nError setting up Neo4j schema: {str(e)}")
        print("\nPlease check:")
        print("1. Your .env file contains correct credentials")
        print("2. The Neo4j database is running and accessible")
        print("3. The URI format is correct (should be: neo4j+s://xxxxx.databases.neo4j.io)")
        print("4. Your username and password are correct")
        raise

if __name__ == "__main__":
    setup_neo4j_schema() 