from typing import List
import os
from dotenv import load_dotenv
import wikipediaapi
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai

# Load environment variables
load_dotenv()

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key":
    raise ValueError(
        "Please set your OpenAI API key in the .env file. "
        "You can get one from https://platform.openai.com/account/api-keys"
    )

class WikipediaGraphRAG:
    def __init__(self):
        # Initialize Wikipedia API
        self.wiki = wikipediaapi.Wikipedia('MyProjectName (contact@example.com)', 'en')
        
        # Initialize Neo4j
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD")
        )
        
        # Initialize OpenAI and ChromaDB
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=384  # smaller dimension for cost efficiency
        )
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Initialize LLM
        self.llm = ChatOpenAI(temperature=0)
        
        # Initialize GraphCypherQAChain
        self.chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            verbose=True,
            return_intermediate_steps=True,
            allow_dangerous_requests=True
        )

    def fetch_wikipedia_content(self, topics: List[str]):
        """Fetch content from Wikipedia for given topics and store in Neo4j and ChromaDB"""
        try:
            for topic in topics:
                page = self.wiki.page(topic)
                if not page.exists():
                    print(f"Page {topic} does not exist")
                    continue
                
                print(f"Processing {topic}...")
                
                # Store in Neo4j with better relationship data
                print(f"Storing {topic} in Neo4j...")
                self.graph.query("""
                    MERGE (t:Topic {name: $topic})
                    SET t.summary = $summary,
                        t.content = $content,
                        t.full_name = $topic
                """, {
                    "topic": topic, 
                    "summary": page.summary, 
                    "content": page.text[:500]
                })
            
            # Create relationships between all topics after storing them
            print("Creating relationships between topics...")
            for i in range(len(topics)):
                for j in range(i + 1, len(topics)):
                    self.graph.query("""
                        MATCH (t1:Topic {name: $topic1})
                        MATCH (t2:Topic {name: $topic2})
                        MERGE (t1)-[r1:LINKS_TO]->(t2)
                        MERGE (t2)-[r2:LINKS_TO]->(t1)
                        SET r1.type = 'related',
                            r2.type = 'related'
                    """, {
                        "topic1": topics[i],
                        "topic2": topics[j]
                    })
                    print(f"Created bidirectional relationship between {topics[i]} and {topics[j]}")
        except openai.RateLimitError as e:
            print("\nOpenAI API Error: Rate limit or quota exceeded.")
            print("Please check your OpenAI billing status at: https://platform.openai.com/account/billing/overview")
            print("Error details:", str(e))
            raise
        except Exception as e:
            print(f"\nError processing content: {str(e)}")
            raise

    def query(self, question: str) -> dict:
        """Query the GraphRAG system"""
        # First, let's check if the topics exist
        topics = self.graph.query("""
            MATCH (t:Topic)
            WHERE t.name IN ['Artificial Intelligence', 'Neural Networks']
            RETURN t.name as name, t.summary as summary
        """)
        
        if not topics:
            print("Warning: One or both topics not found in database")
        
        result = self.chain.invoke(question)
        
        # Extract intermediate steps
        intermediate_steps = result.get("intermediate_steps", [])
        cypher_query = ""
        context = ""
        
        if intermediate_steps:
            # Use a more flexible query that includes both direct and indirect relationships
            cypher_query = """
                MATCH (t1:Topic {name: 'Artificial Intelligence'})
                MATCH (t2:Topic {name: 'Neural Networks'})
                OPTIONAL MATCH path = shortestPath((t1)-[:LINKS_TO*]-(t2))
                RETURN 
                    t1.summary as ai_summary,
                    t2.summary as nn_summary,
                    t1.content as ai_content,
                    t2.content as nn_content
            """
            
            # Execute our custom query
            relationship_data = self.graph.query(cypher_query)
            if relationship_data:
                context = (
                    f"AI Summary: {relationship_data[0]['ai_summary']}\n\n"
                    f"Neural Networks Summary: {relationship_data[0]['nn_summary']}\n\n"
                    f"AI Content: {relationship_data[0]['ai_content'][:200]}...\n\n"
                    f"Neural Networks Content: {relationship_data[0]['nn_content'][:200]}..."
                )
            else:
                context = "No relationship found between the topics"
        
        return {
            "answer": result["result"],
            "cypher_query": cypher_query,
            "context": context
        }

    def clear_database(self):
        """Clear all data from Neo4j and ChromaDB"""
        print("Clearing Neo4j database...")
        self.graph.query("""
            MATCH (n)
            DETACH DELETE n
        """)
        
        print("Clearing ChromaDB...")
        self.vectorstore.delete_collection()
        
    def get_stored_topics(self):
        """Get list of topics already stored in Neo4j"""
        result = self.graph.query("""
            MATCH (t:Topic)
            RETURN t.name as name
        """)
        return [record["name"] for record in result]

def main():
    # Initialize the system
    rag_system = WikipediaGraphRAG()
    
    # Clear existing data
    # print("Clearing existing data...")
    # rag_system.clear_database()
    
    # Show existing topics
    existing_topics = rag_system.get_stored_topics()
    if existing_topics:
        print(f"Already stored topics: {existing_topics}")
    else:
        print("No topics stored in database yet.")
    
    # Example usage: fetch some related topics
    topics = ["Artificial Intelligence", "Machine Learning", "Neural Networks"]
    print(f"\nProcessing topics: {topics}")
    rag_system.fetch_wikipedia_content(topics)
    
    # Example query
    question = "What is the relationship between Artificial Intelligence and Neural Networks?"
    print(f"\nQuerying: {question}")
    result = rag_system.query(question)
    print(f"\nAnswer: {result['answer']}")
    print(f"\nCypher Query Used: {result['cypher_query']}")
    print(f"\nContext Retrieved: {result['context']}")

if __name__ == "__main__":
    main() 