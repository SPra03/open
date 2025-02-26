from typing import List, Dict, Any
import os
import re
import hashlib
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "Please set your OpenAI API key in the .env file for the Pharmaceutical RAG application. "
        "You can get one from https://platform.openai.com/account/api-keys"
    )

class PharmaGraphRAG:
    def __init__(self):
        # Initialize Neo4j
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD")
        )
        
        # Setup schema for pharmaceutical data
        self._setup_schema()
        
        # Initialize OpenAI
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=384
        )
        
        # Initialize vector store
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory="./chroma_pharma_db"
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

    def _setup_schema(self):
        """Create schema for pharmaceutical insights graph"""
        # Create constraints for unique IDs
        self.graph.query("""
            CREATE CONSTRAINT sr_id IF NOT EXISTS
            FOR (sr:SalesRep) REQUIRE sr.id IS UNIQUE
        """)
        
        self.graph.query("""
            CREATE CONSTRAINT hcp_name IF NOT EXISTS
            FOR (h:HCP) REQUIRE h.name IS UNIQUE
        """)
        
        self.graph.query("""
            CREATE CONSTRAINT product_name IF NOT EXISTS
            FOR (p:Product) REQUIRE p.name IS UNIQUE
        """)
        
        self.graph.query("""
            CREATE CONSTRAINT visit_id IF NOT EXISTS
            FOR (v:Visit) REQUIRE v.id IS UNIQUE
        """)
        
        self.graph.query("""
            CREATE CONSTRAINT insight_id IF NOT EXISTS
            FOR (i:Insight) REQUIRE i.id IS UNIQUE
        """)
        
        # Create indexes for performance
        self.graph.query("""
            CREATE INDEX visit_date IF NOT EXISTS FOR (v:Visit) ON (v.date)
        """)
        
        self.graph.query("""
            CREATE INDEX insight_type IF NOT EXISTS FOR (i:Insight) ON (i.type)
        """)
        
        self.graph.query("""
            CREATE INDEX product_name_idx IF NOT EXISTS FOR (p:Product) ON (p.name)
        """)

    def _extract_hcp_name(self, text):
        """Extract HCP name from text using regex"""
        pattern = r"Dr\.\s+([A-Za-z]+)"
        match = re.search(pattern, text)
        if match:
            return f"Dr. {match.group(1)}"
        return "Unknown HCP"

    def _extract_visit_date(self, text):
        """Extract visit date from text using regex"""
        pattern = r"(\d{2}/\d{2})"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        return "Unknown Date"

    def _extract_key_points(self, text, max_points=3):
        """Extract key points from text using simple sentence splitting"""
        # Simple approach: just take first few sentences
        sentences = re.split(r'[.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences[:max_points]

    def process_pharma_data(self, csv_path):
        """Process pharmaceutical data from CSV and build knowledge graph"""
        # First verify the database is empty
        count = self.graph.query("""
            MATCH (n)
            RETURN count(n) as count
        """)
        
        if count[0]['count'] > 0:
            print(f"WARNING: Database contains {count[0]['count']} nodes. Consider clearing it first.")
        
        # Initialize OpenAI for insight comparison
        insight_analyzer = ChatOpenAI(temperature=0)
        
        # Read CSV data
        df = pd.read_csv(csv_path)
        
        # Process each interaction
        for _, row in df.iterrows():
            # Extract base data
            rep_id = row['Sales Rep ID']
            rep_name = row['Sales Rep Name']
            product = row['Product to Promote to Healthcare Provider']
            xai_insight = row['XAI Insights Based on Model']
            field_insight = row['Field Insights Provided by Sales Rep']
            
            # Extract entities
            hcp_name = self._extract_hcp_name(xai_insight)
            visit_date = self._extract_visit_date(field_insight)
            
            # Generate unique IDs
            visit_id = hashlib.md5(f"{rep_id}_{hcp_name}_{visit_date}_{product}".encode()).hexdigest()
            xai_id = f"xai_{visit_id}"
            field_id = f"field_{visit_id}"
            
            # Extract key points
            xai_key_points = self._extract_key_points(xai_insight)
            field_key_points = self._extract_key_points(field_insight)
            
            # Perform advanced insight comparison using LLM
            comparison_prompt = f"""
            Compare these two insights about {product} for {hcp_name} and determine their relationship:
            
            XAI INSIGHT: {xai_insight}
            
            FIELD INSIGHT: {field_insight}
            
            Analyze if the field insight CONFIRMS, CONTRADICTS, or EXTENDS the XAI insight.
            Also identify key similarities and differences.
            
            Return your analysis in this JSON format:
            {{
                "relationship": "CONFIRMS|CONTRADICTS|EXTENDS",
                "confidence": 0.0-1.0,
                "similarities": ["similarity1", "similarity2"],
                "differences": ["difference1", "difference2"],
                "summary": "brief explanation of the relationship"
            }}
            """
            
            try:
                comparison_result = insight_analyzer.invoke(comparison_prompt)
                import json
                analysis = json.loads(comparison_result.content)
                
                # Extract relationship and properties
                insight_relationship = analysis["relationship"]
                relationship_properties = {
                    "confidence": analysis["confidence"],
                    "summary": analysis["summary"],
                    "similarities": json.dumps(analysis["similarities"]),
                    "differences": json.dumps(analysis["differences"])
                }
            except Exception as e:
                print(f"Error analyzing insights: {str(e)}")
                insight_relationship = "RELATES_TO"
                relationship_properties = {
                    "confidence": 0.5,
                    "summary": "Automated analysis failed",
                    "similarities": "[]",
                    "differences": "[]"
                }
            
            # Create the graph structure
            self.graph.query("""
                // Create nodes
                MERGE (sr:SalesRep {id: $rep_id})
                SET sr.name = $rep_name
                
                MERGE (hcp:HCP {name: $hcp_name})
                
                MERGE (product:Product {name: $product})
                
                MERGE (visit:Visit {id: $visit_id})
                SET visit.date = $visit_date
                
                MERGE (xai:Insight {id: $xai_id})
                SET xai.type = "XAI",
                    xai.content = $xai_insight,
                    xai.key_points = $xai_key_points
                
                MERGE (field:Insight {id: $field_id})
                SET field.type = "Field",
                    field.content = $field_insight,
                    field.key_points = $field_key_points
                
                // Return nodes for relationship creation
                RETURN sr, hcp, product, visit, xai, field
            """, {
                "rep_id": rep_id,
                "rep_name": rep_name,
                "hcp_name": hcp_name,
                "product": product,
                "visit_id": visit_id,
                "visit_date": visit_date,
                "xai_id": xai_id,
                "xai_insight": xai_insight,
                "xai_key_points": xai_key_points,
                "field_id": field_id,
                "field_insight": field_insight,
                "field_key_points": field_key_points
            })
            
            # Create relationships in separate queries to ensure they work
            self.graph.query("""
                MATCH (sr:SalesRep {id: $rep_id})
                MATCH (v:Visit {id: $visit_id})
                CREATE (sr)-[:CONDUCTED]->(v)
            """, {"rep_id": rep_id, "visit_id": visit_id})
            
            self.graph.query("""
                MATCH (v:Visit {id: $visit_id})
                MATCH (hcp:HCP {name: $hcp_name})
                CREATE (v)-[:WITH]->(hcp)
            """, {"visit_id": visit_id, "hcp_name": hcp_name})
            
            self.graph.query("""
                MATCH (v:Visit {id: $visit_id})
                MATCH (p:Product {name: $product})
                CREATE (v)-[:DISCUSSED]->(p)
            """, {"visit_id": visit_id, "product": product})
            
            self.graph.query("""
                MATCH (v:Visit {id: $visit_id})
                MATCH (xai:Insight {id: $xai_id})
                CREATE (v)-[:GENERATED]->(xai)
            """, {"visit_id": visit_id, "xai_id": xai_id})
            
            self.graph.query("""
                MATCH (v:Visit {id: $visit_id})
                MATCH (i:Insight {id: $field_id})
                CREATE (v)-[:RESULTED_IN]->(i)
            """, {"visit_id": visit_id, "field_id": field_id})
            
            self.graph.query("""
                MATCH (field:Insight {id: $field_id})
                MATCH (xai:Insight {id: $xai_id})
                CREATE (field)-[r:RELATES_TO]->(xai)
                SET r.confidence = $confidence,
                    r.summary = $summary,
                    r.similarities = $similarities,
                    r.differences = $differences,
                    r.relationship_type = $relationship_type
            """, {
                "field_id": field_id, 
                "xai_id": xai_id, 
                "relationship_type": insight_relationship,
                "confidence": relationship_properties["confidence"],
                "summary": relationship_properties["summary"],
                "similarities": relationship_properties["similarities"],
                "differences": relationship_properties["differences"]
            })
            
            # Store insights in vector db for semantic search
            metadata = {
                "type": "xai_insight",
                "product": product,
                "hcp": hcp_name,
                "rep_id": rep_id,
                "rep_name": rep_name,
                "visit_date": visit_date
            }
            self.vectorstore.add_texts(texts=[xai_insight], metadatas=[metadata])
            
            metadata = {
                "type": "field_insight",
                "product": product,
                "hcp": hcp_name,
                "rep_id": rep_id,
                "rep_name": rep_name,
                "visit_date": visit_date
            }
            self.vectorstore.add_texts(texts=[field_insight], metadatas=[metadata])
            
        # Print summary of loaded data
        summary = self.graph.query("""
            MATCH (sr:SalesRep) 
            RETURN count(DISTINCT sr) as rep_count
        """)
        rep_count = summary[0]["rep_count"]
        
        summary = self.graph.query("""
            MATCH (p:Product) 
            RETURN count(DISTINCT p) as product_count
        """)
        product_count = summary[0]["product_count"]
        
        summary = self.graph.query("""
            MATCH (h:HCP) 
            RETURN count(DISTINCT h) as hcp_count
        """)
        hcp_count = summary[0]["hcp_count"]
        
        summary = self.graph.query("""
            MATCH (v:Visit) 
            RETURN count(v) as visit_count
        """)
        visit_count = summary[0]["visit_count"]
        
        print(f"Loaded: {rep_count} Sales Reps, {product_count} Products, {hcp_count} HCPs, {visit_count} Visits")
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the pharmaceutical knowledge graph"""
        # Check if this is a common query pattern we can optimize
        
        # First check for product names in the question
        products = self.get_products()
        found_product = None
        
        for product in products:
            # Extract base product name without the generic name in parentheses
            base_product_name = product.split(" (")[0] if " (" in product else product
            if base_product_name.lower() in question.lower():
                found_product = product
                break
        
        # If we found a product, check if this is a comparison question
        if found_product and ("xai" in question.lower() and "compare" in question.lower()):
            print(f"Found comparison question for product: {found_product}")
            return self._comparison_query(found_product, question)
        elif found_product:
            return self._product_query(found_product, question)
        
        # Check for comparison questions (original approach as fallback)
        comparison_patterns = [
            r"compare .* xai .* field .* (.*)\??",
            r"how do .* xai .* (?:predictions |insights )?compare .* (.*)\??",
            r"relationship .* xai .* field .* (.*)\??"
        ]
        
        print(f"Checking comparison patterns for: {question}")
        for pattern in comparison_patterns:
            match = re.search(pattern, question.lower())
            if match:
                print(f"Matched pattern: {pattern}")
                # Handle the direct pattern separately
                if "cardiofix" in pattern:
                    product_name = "cardiofix"
                else:
                    print(f"Product name: {match.group(1).strip()}")
                    product_name = match.group(1).strip()
                
                matched_product = self._find_matching_product(product_name)
                print(f"Matched product: {matched_product}")
                if matched_product:
                    return self._comparison_query(matched_product, question)
        
        # First check for exact product names in the question
        products = self.get_products()
        for product in products:
            # Extract base product name without the generic name in parentheses
            base_product_name = product.split(" (")[0] if " (" in product else product
            if base_product_name.lower() in question.lower():
                return self._product_query(product, question)
        
        # Check for product-specific questions with regex patterns
        product_patterns = [
            r"what .* say .* about ([A-Za-z]+)\??",
            r"insights .* for ([A-Za-z]+)\??",
            r"feedback .* on ([A-Za-z]+)\??",
            r"([A-Za-z]+) .* feedback\??",
            r"([A-Za-z]+) .* insights\??"
        ]
        
        for pattern in product_patterns:
            match = re.search(pattern, question.lower())
            if match:
                product_name = match.group(1).capitalize()
                # Look for partial matches in product names
                for product in products:
                    base_product_name = product.split(" (")[0] if " (" in product else product
                    if product_name.lower() == base_product_name.lower():
                        return self._product_query(product, question)
        
        # Use the LLM to generate a Cypher query
        cypher_query = None
        answer = None
        context = None
        
        try:
            result = self.chain.invoke(question)
            answer = result["result"]
            context = str(result.get("intermediate_steps", "No steps available"))
        except Exception as e:
            answer = f"I encountered an error while processing your question: {str(e)}"
            context = "Error occurred during query processing"
        
        return {
            "answer": answer,
            "cypher_query": cypher_query,
            "context": context
        }
        
    def _product_query(self, product_name: str, original_question: str) -> Dict[str, Any]:
        """Handle product-specific queries with optimized Cypher"""
        # Direct Cypher query for product insights
        cypher_query = """
            MATCH (p:Product)
            WHERE toLower(p.name) = toLower($product_name)
            MATCH (v:Visit)-[:DISCUSSED]->(p)
            MATCH (v)-[:RESULTED_IN]->(field:Insight {type: "Field"})
            MATCH (sr:SalesRep)-[:CONDUCTED]->(v)
            RETURN sr.name as rep_name, field.content as insight, v.date as date
            ORDER BY v.date DESC
        """
        
        results = self.graph.query(cypher_query, {"product_name": product_name})
        
        if not results:
            return {
                "answer": f"I don't have any field insights about {product_name} in my database.",
                "cypher_query": cypher_query,
                "context": f"No results found for product: {product_name}"
            }
        
        # Format the results into a coherent answer
        answer = f"Here's what field representatives have reported about {product_name}:\n\n"
        
        for result in results:
            answer += f"- {result['rep_name']} reported: \"{result['insight']}\"\n\n"
        
        return {
            "answer": answer,
            "cypher_query": cypher_query,
            "context": f"Found {len(results)} insights for {product_name}"
        }
    
    def get_products(self):
        """Get list of products in the database"""
        results = self.graph.query("""
            MATCH (p:Product)
            RETURN p.name as name
        """)
        return [r["name"] for r in results]
    
    def get_hcps(self):
        """Get list of HCPs in the database"""
        results = self.graph.query("""
            MATCH (h:HCP)
            RETURN h.name as name
        """)
        return [r["name"] for r in results]
    
    def get_reps(self):
        """Get list of sales reps in the database"""
        results = self.graph.query("""
            MATCH (sr:SalesRep)
            RETURN sr.name as name
        """)
        return [r["name"] for r in results]
    
    def clear_database(self):
        """Clear all data from the database"""
        self.graph.query("""
            MATCH (n)
            DETACH DELETE n
        """)
        
        # Also clear the vector store
        try:
            self.vectorstore.delete_collection()
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory="./chroma_pharma_db"
            )
        except:
            print("Vector store already empty or error clearing it")

    def _find_matching_product(self, product_name: str) -> str:
        """Find a matching product in the database, handling variations in naming"""
        products = self.get_products()
        
        # First try exact match
        for product in products:
            if product.lower() == product_name.lower():
                return product
        
        # Try base name match (without generic name in parentheses)
        for product in products:
            base_product_name = product.split(" (")[0] if " (" in product else product
            if base_product_name.lower() == product_name.lower():
                return product
        
        # Try partial match
        for product in products:
            if product_name.lower() in product.lower() or product.lower() in product_name.lower():
                return product
        
        return None 

    def analyze_insight_relationships(self, product_name=None):
        """Analyze the relationships between XAI and field insights"""
        query_params = {}
        product_filter = ""
        
        if product_name:
            product_filter = "WHERE p.name = $product_name"
            query_params["product_name"] = product_name
        
        # Get relationship statistics
        cypher_query = f"""
            MATCH (field:Insight {{type: "Field"}})-[r:RELATES_TO]->(xai:Insight {{type: "XAI"}})
            MATCH (v:Visit)-[:RESULTED_IN]->(field)
            MATCH (v)-[:DISCUSSED]->(p:Product)
            {product_filter}
            RETURN r.relationship_type as relationship, 
                   count(*) as count,
                   avg(r.confidence) as avg_confidence,
                   p.name as product
            ORDER BY count DESC
        """
        
        relationship_stats = self.graph.query(cypher_query, query_params)
        
        # Get top contradictions
        contradictions_query = f"""
            MATCH (field:Insight {{type: "Field"}})-[r:RELATES_TO]->(xai:Insight {{type: "XAI"}})
            WHERE r.relationship_type = 'CONTRADICTS'
            MATCH (v:Visit)-[:RESULTED_IN]->(field)
            MATCH (v)-[:DISCUSSED]->(p:Product)
            {product_filter}
            RETURN p.name as product,
                   field.content as field_insight,
                   xai.content as xai_insight,
                   r.summary as summary,
                   r.confidence as confidence
            ORDER BY r.confidence DESC
            LIMIT 5
        """
        
        contradictions = self.graph.query(contradictions_query, query_params)
        
        return {
            "stats": relationship_stats,
            "contradictions": contradictions
        } 

    def _comparison_query(self, product_name: str, original_question: str) -> Dict[str, Any]:
        """Handle XAI vs field insight comparison queries"""
        # Get the relationship statistics
        insight_analysis = self.analyze_insight_relationships(product_name)
        
        # Get specific examples of each relationship type
        examples_query = """
            MATCH (field:Insight {type: "Field"})-[r:RELATES_TO]->(xai:Insight {type: "XAI"})
            MATCH (v:Visit)-[:RESULTED_IN]->(field)
            MATCH (v)-[:DISCUSSED]->(p:Product {name: $product_name})
            RETURN r.relationship_type as relationship,
                   field.content as field_insight,
                   xai.content as xai_insight,
                   r.summary as summary,
                   r.confidence as confidence
            ORDER BY r.confidence DESC
            LIMIT 5
        """
        
        examples = self.graph.query(examples_query, {"product_name": product_name})
        
        # Format the answer
        answer = f"# XAI vs Field Insights Comparison for {product_name}\n\n"
        
        # Add relationship overview
        if insight_analysis["stats"]:
            answer += "## Relationship Overview\n\n"
            for stat in insight_analysis["stats"]:
                if stat["relationship"] == "CONFIRMS":
                    answer += f"- **{stat['count']}** field insights **confirm** XAI predictions\n"
                elif stat["relationship"] == "CONTRADICTS":
                    answer += f"- **{stat['count']}** field insights **contradict** XAI predictions\n"
                elif stat["relationship"] == "EXTENDS":
                    answer += f"- **{stat['count']}** field insights **extend** XAI predictions\n"
            answer += "\n"
        
        # Add examples
        if examples:
            answer += "## Examples\n\n"
            for example in examples:
                answer += f"### {example['relationship']} (Confidence: {example['confidence']:.2f})\n\n"
                answer += f"**XAI Insight:** {example['xai_insight']}\n\n"
                answer += f"**Field Insight:** {example['field_insight']}\n\n"
                answer += f"**Analysis:** {example['summary']}\n\n"
        
        return {
            "answer": answer,
            "cypher_query": examples_query,
            "context": f"Analyzed {len(insight_analysis['stats'])} relationship types and {len(examples)} examples"
        } 