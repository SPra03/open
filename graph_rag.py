from typing import List
import os
from dotenv import load_dotenv
import wikipediaapi
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai

# TODO(sourabh): Add error handling for API key
if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key":
    raise ValueError(
        "Please set your OpenAI API key in the .env file. "
        "You can get one from https://platform.openai.com/account/api-keys"
    )

class WikipediaGraphRAG:
    def __init__(self):
        # Initialize Wikipedia API
        self.wiki = wikipediaapi.Wikipedia('MyProjectName (contact@example.com)', 'en')
        
        # FIXME: Increase chunk size if memory allows
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Initialize Neo4j
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD")
        )
        
        # Create constraints and indexes if they don't exist
        self._setup_schema()
        
        # Initialize OpenAI and ChromaDB
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=384  # hack: smaller dims = lower cost
        )
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        # Initialize LLM
        self.llm = ChatOpenAI(temperature=0)
        
        # XXX: dangerous_requests needed for custom Cypher
        self.chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            verbose=True,
            return_intermediate_steps=True,
            allow_dangerous_requests=True
        )

    def _setup_schema(self):
        """Setup Neo4j schema with constraints and indexes"""
        self.graph.query("""
            CREATE CONSTRAINT topic_name IF NOT EXISTS
            FOR (t:Topic) REQUIRE t.name IS UNIQUE
        """)
        
        self.graph.query("""
            CREATE INDEX topic_category_idx IF NOT EXISTS
            FOR (t:Topic) ON (t.category)
        """)

    def fetch_wikipedia_content(self, topics: List[str]):
        """Fetch content from Wikipedia for given topics and store in Neo4j and ChromaDB"""
        try:
            # First pass: Store all topics and collect their links
            topic_links = {}
            for topic in topics:
                page = self.wiki.page(topic)
                if not page.exists():
                    print(f"Page {topic} does not exist")
                    continue
                
                print(f"Processing {topic}...")
                
                # Get categories and references
                categories = [cat.split(':')[-1] for cat in page.categories]
                references = [ref for ref in page.references]
                
                # Collect all links from the page
                links = []
                for link_title in page.links:
                    link_page = self.wiki.page(link_title)
                    if link_page.exists():
                        links.append(link_title)
                topic_links[topic] = links
                
                # Get sections
                sections = []
                for section in page.sections:
                    if section.text:
                        sections.append({
                            'title': section.title,
                            'content': section.text[:500]  # First 500 chars of each section
                        })
                
                # Store in Neo4j with better relationship data
                print(f"Storing {topic} in Neo4j...")
                self.graph.query("""
                    MERGE (t:Topic {name: $topic})
                    SET t.summary = $summary,
                        t.content = $content,
                        t.categories = $categories,
                        t.last_updated = datetime(),
                        t.section_count = $section_count,
                        t.reference_count = $reference_count,
                        t.link_count = $link_count,
                        t.full_name = $topic
                """, {
                    "topic": topic, 
                    "summary": page.summary, 
                    "content": page.text[:500],
                    "categories": categories,
                    "section_count": len(sections),
                    "reference_count": len(references),
                    "link_count": len(links)
                })
                
                # Store sections as separate nodes
                for section in sections:
                    self.graph.query("""
                        MATCH (t:Topic {name: $topic})
                        MERGE (s:Section {
                            title: $title,
                            topic: $topic
                        })
                        SET s.content = $content
                        MERGE (t)-[r:HAS_SECTION]->(s)
                    """, {
                        "topic": topic,
                        "title": section['title'],
                        "content": section['content']
                    })
            
            # Second pass: Create relationships based on various factors
            print("Creating automated relationships...")
            self.graph.query("""
                // First, create relationships based on direct Wikipedia links
                MATCH (t1:Topic)
                WHERE t1.name IN $all_topics
                MATCH (t2:Topic)
                WHERE t2.name IN $all_topics AND t1 <> t2
                WITH t1, t2
                
                // Calculate relationship strength based on multiple factors
                OPTIONAL MATCH (t1)-[existing:LINKS_TO]-(t2)
                WITH t1, t2, existing,
                     // Check if they share categories
                     [cat IN t1.categories WHERE cat IN t2.categories] as shared_cats,
                     // Check if one links to the other
                     CASE WHEN t2.name IN $topic_links[t1.name] THEN 1 ELSE 0 END as direct_link,
                     // Check for common links
                     size([link IN $topic_links[t1.name] WHERE link IN $topic_links[t2.name]]) as common_links
                
                // Calculate overall relationship strength
                WITH t1, t2, existing, shared_cats,
                     (size(shared_cats) * 2 + direct_link * 3 + common_links) as strength
                WHERE strength > 0
                
                // Create or update relationship
                MERGE (t1)-[r:LINKS_TO]->(t2)
                SET r.type = 'automated',
                    r.shared_categories = shared_cats,
                    r.strength = strength,
                    r.has_direct_link = direct_link = 1,
                    r.common_links_count = common_links,
                    r.last_updated = datetime()
            """, {
                "all_topics": list(topic_links.keys()),
                "topic_links": topic_links
            })
            
            # Print relationship summary
            relationships = self.graph.query("""
                MATCH (t1:Topic)-[r:LINKS_TO]->(t2:Topic)
                RETURN t1.name as source, t2.name as target, 
                       r.strength as strength, 
                       r.has_direct_link as direct_link
                ORDER BY r.strength DESC
            """)
            
            print("\nAutomated Relationships Created:")
            for rel in relationships:
                print(f"{rel['source']} → {rel['target']} (Strength: {rel['strength']})")
        except openai.RateLimitError as e:
            print("\nOpenAI API Error: Rate limit or quota exceeded.")
            print("Please check your OpenAI billing status at: https://platform.openai.com/account/billing/overview")
            print("Error details:", str(e))
            raise
        except Exception as e:
            print(f"\nError processing content: {str(e)}")
            raise

    def query(self, question: str, context_topic: str = None) -> dict:
        """Query the GraphRAG system"""
        # NOTE: keep this mapping updated when adding new topics
        topic_aliases = {
            "ai": "Artificial Intelligence",
            "ml": "Machine Learning", 
            "nn": "Neural Networks",
            "artificial intelligence": "Artificial Intelligence",
            "machine learning": "Machine Learning",
            "neural networks": "Neural Networks",
            "neural network": "Neural Networks",
            "deep learning": "Deep Learning"
        }
        
        relationship_keywords = ["related", "relationship", "between", "connection", "and"]
        is_relationship_question = any(keyword in question.lower() for keyword in relationship_keywords)
        
        if is_relationship_question:
            # Find all mentioned topics
            mentioned_topics = []
            for alias, full_name in topic_aliases.items():
                if alias in question.lower():
                    mentioned_topics.append(full_name)
            
            if len(mentioned_topics) >= 2:
                # Query for relationship between topics
                cypher_query = """
                MATCH (t1:Topic {name: $topic1})
                MATCH (t2:Topic {name: $topic2})
                OPTIONAL MATCH (t1)-[r:LINKS_TO]-(t2)
                RETURN 
                    t1.name as topic1_name,
                    t1.summary as topic1_summary,
                    t2.name as topic2_name,
                    t2.summary as topic2_summary,
                    COALESCE(r.strength, 0) as relationship_strength,
                    COALESCE(r.shared_categories, []) as shared_categories
                """
                
                relationship_data = self.graph.query(cypher_query, {
                    "topic1": mentioned_topics[0],
                    "topic2": mentioned_topics[1]
                })
                
                if relationship_data:
                    data = relationship_data[0]
                    context = (
                        f"{data['topic1_name']}:\n{data['topic1_summary']}\n\n"
                        f"{data['topic2_name']}:\n{data['topic2_summary']}\n\n"
                        f"Relationship Strength: {data['relationship_strength']}\n"
                        f"Shared Categories: {', '.join(data['shared_categories'] or [])}"
                    )
                    
                    # Construct a meaningful answer about the relationship
                    answer = (
                        f"{data['topic1_name']} and {data['topic2_name']} are related. "
                        f"{data['topic1_name']} is {data['topic1_summary'][:100]}... "
                        f"while {data['topic2_name']} is {data['topic2_summary'][:100]}..."
                    )
                    
                    return {
                        "answer": answer,
                        "cypher_query": cypher_query,
                        "context": context
                    }
        
        # Initialize variables
        topics = None
        cypher_query = ""
        context = ""
        
        # Check if database is empty
        stored_topics = self.get_stored_topics()
        if not stored_topics:
            return {
                "answer": "No topics found in database. Please add some topics first.",
                "cypher_query": "",
                "context": ""
            }
        
        # Extract potential topic from question
        question_lower = question.lower()
        potential_topic = None
        
        # Check for direct mentions of topics or their aliases
        for word in question_lower.replace("?", "").split():
            normalized = topic_aliases.get(word, word)
            if normalized in stored_topics:
                potential_topic = normalized
                break
        
        # If no direct match, try common patterns
        if not potential_topic:
            for alias, full_name in topic_aliases.items():
                if alias in question_lower:
                    potential_topic = full_name
                    break
        
        # If question is about a specific topic (like "What is AI?")
        simple_topic_keywords = ["what is", "what are", "tell me about", "explain"]
        if any(keyword in question_lower for keyword in simple_topic_keywords):
            context_topic = potential_topic
            
            if context_topic:
                cypher_query = """
                    MATCH (t:Topic {name: $topic})
                    OPTIONAL MATCH (t)-[:LINKS_TO]-(related:Topic)
                    RETURN 
                        t.summary as main_summary,
                        t.content as main_content,
                        collect(DISTINCT related.name) as related_topics
                """
                
                topic_data = self.graph.query(cypher_query, {"topic": context_topic})
                if topic_data:
                    context = (
                        f"{context_topic}:\n"
                        f"Summary: {topic_data[0]['main_summary']}\n\n"
                        f"Content: {topic_data[0]['main_content'][:300]}...\n\n"
                        f"Related Topics: {', '.join(topic_data[0]['related_topics'])}"
                    )
                    return {
                        "answer": topic_data[0]['main_summary'],
                        "cypher_query": cypher_query,
                        "context": context
                    }
                else:
                    context = f"No information found for topic: {context_topic}"
        
        if not topics and context_topic:
            print(f"Warning: Topic {context_topic} not found in database")
        
        result = self.chain.invoke(question)
        
        # Extract intermediate steps
        intermediate_steps = result.get("intermediate_steps", [])
        
        if intermediate_steps and not context:  # Only if context wasn't set by relationship query
            # Use a more flexible query that includes both direct and indirect relationships
            cypher_query = """
                MATCH (t:Topic {name: $topic})
                OPTIONAL MATCH (t)-[:LINKS_TO*1..2]-(related:Topic)
                WHERE related.name <> $topic
                RETURN 
                    t.summary as main_summary,
                    t.content as main_content,
                    collect(DISTINCT related.name) as related_topics,
                    collect(DISTINCT related.summary) as related_summaries
            """
            
            # Execute our custom query
            relationship_data = self.graph.query(cypher_query, {"topic": context_topic})
            if relationship_data:
                context = (
                    f"Main Topic ({context_topic}):\n"
                    f"Summary: {relationship_data[0]['main_summary']}\n\n"
                    f"Content: {relationship_data[0]['main_content'][:200]}...\n\n"
                    f"Related Topics: {', '.join(relationship_data[0]['related_topics'])}\n\n"
                    f"Related Summaries:\n" +
                    "\n".join([f"- {s[:200]}..." for s in relationship_data[0]['related_summaries']])
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
    
    # Example usage: fetch core topics with common aliases
    topics = ["Artificial Intelligence", "Machine Learning", "Neural Networks", "Deep Learning"]
    print(f"\nProcessing topics: {topics}")
    rag_system.fetch_wikipedia_content(topics)
    
    # Test different types of questions
    questions = [
        "What is AI?",
        "Tell me about Machine Learning",
        "What is the relationship between AI and Neural Networks?",
    ]
    
    for question in questions:
        print(f"\nQuerying: {question}")
        result = rag_system.query(question)
        print(f"\nAnswer: {result['answer']}")
        print(f"\nContext Retrieved: {result['context']}")

if __name__ == "__main__":
    main() 