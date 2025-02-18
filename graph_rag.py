from typing import List
import os
from dotenv import load_dotenv
import wikipediaapi
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.graphs import Neo4jGraph
from langchain.chains import GraphRAGChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

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
        self.embeddings = OpenAIEmbeddings()
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
        
        # Initialize GraphRAG chain
        self.chain = GraphRAGChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            vectorstore=self.vectorstore,
            verbose=True
        )

    def fetch_wikipedia_content(self, topics: List[str]):
        """Fetch content from Wikipedia for given topics and store in Neo4j and ChromaDB"""
        for topic in topics:
            page = self.wiki.page(topic)
            if not page.exists():
                print(f"Page {topic} does not exist")
                continue
            
            # Split content into chunks
            chunks = self.text_splitter.split_text(page.text)
            
            # Store in ChromaDB
            self.vectorstore.add_texts(
                texts=chunks,
                metadatas=[{"source": topic, "chunk": i} for i in range(len(chunks))]
            )
            
            # Store in Neo4j
            # Create node for the topic
            self.graph.query("""
                MERGE (t:Topic {name: $topic})
                SET t.summary = $summary
            """, {"topic": topic, "summary": page.summary})
            
            # Store relationships with other topics (links)
            for link in page.links:
                self.graph.query("""
                    MATCH (t1:Topic {name: $topic1})
                    MERGE (t2:Topic {name: $topic2})
                    MERGE (t1)-[r:LINKS_TO]->(t2)
                """, {"topic1": topic, "topic2": link})

    def query(self, question: str) -> str:
        """Query the GraphRAG system"""
        return self.chain.run(question)

def main():
    # Initialize the system
    rag_system = WikipediaGraphRAG()
    
    # Example usage: fetch some related topics
    topics = ["Artificial Intelligence", "Machine Learning", "Neural Networks"]
    rag_system.fetch_wikipedia_content(topics)
    
    # Example query
    question = "What is the relationship between AI and Neural Networks?"
    answer = rag_system.query(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main() 