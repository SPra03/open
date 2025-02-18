from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

def setup_neo4j_schema():
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    )
    
    with driver.session() as session:
        # Create constraints
        session.run("""
            CREATE CONSTRAINT topic_name IF NOT EXISTS
            FOR (t:Topic) REQUIRE t.name IS UNIQUE
        """)
        
        # Create indexes
        session.run("""
            CREATE INDEX topic_name_idx IF NOT EXISTS
            FOR (t:Topic) ON (t.name)
        """)
    
    driver.close()

if __name__ == "__main__":
    setup_neo4j_schema() 