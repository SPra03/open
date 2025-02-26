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
            
            # Determine insight relationship (simplified version)
            # In a real app, you might use sentiment analysis or more sophisticated comparison
            if "receptive" in field_insight.lower() or "interest" in field_insight.lower():
                insight_relationship = "CONFIRMS"
            elif "concern" in field_insight.lower() or "issue" in field_insight.lower():
                insight_relationship = "CONTRADICTS"
            else:
                insight_relationship = "RELATES_TO"
            
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
                
                // Create relationships
                MERGE (sr)-[:CONDUCTED]->(visit)
                MERGE (visit)-[:WITH]->(hcp)
                MERGE (visit)-[:DISCUSSED]->(product)
                MERGE (visit)-[:GENERATED]->(xai)
                MERGE (visit)-[:RESULTED_IN]->(field)
                MERGE (field)-[r:RELATES_TO]->(xai)
                SET r.type = $insight_relationship
            """, {
                "rep_id": rep_id,
                "rep_name": rep_name,
                "hcp_name": hcp_name,
                "product": product,
                "visit_id": visit_id,
                "visit_date": visit_date,
                "xai_id": xai_id,
                "field_id": field_id,
                "xai_insight": xai_insight,
                "field_insight": field_insight,
                "xai_key_points": xai_key_points,
                "field_key_points": field_key_points,
                "insight_relationship": insight_relationship
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
    
    def query(self, question: str) -> dict:
        """Query the pharmaceutical RAG system"""
        # Detect query type
        products = self.get_products()
        product_mentioned = None
        for product in products:
            if product.lower() in question.lower():
                product_mentioned = product
                break
        
        hcps = self.get_hcps()
        hcp_mentioned = None
        for hcp in hcps:
            if hcp.lower() in question.lower():
                hcp_mentioned = hcp
                break
        
        reps = self.get_reps()
        rep_mentioned = None
        for rep_name in reps:
            if rep_name.lower() in question.lower():
                rep_mentioned = rep_name
                break
        
        # Handle different types of queries
        cypher_query = ""
        context = ""
        
        # Product-specific queries
        if product_mentioned and "field" in question.lower():
            # Get field insights for a specific product
            cypher_query = """
                MATCH (p:Product {name: $product})
                MATCH (v:Visit)-[:DISCUSSED]->(p)
                MATCH (v)-[:RESULTED_IN]->(i:Insight {type: "Field"})
                MATCH (sr:SalesRep)-[:CONDUCTED]->(v)-[:WITH]->(hcp:HCP)
                RETURN sr.name as rep, hcp.name as hcp, v.date as date, 
                       i.content as insight, p.name as product
                ORDER BY v.date DESC
            """
            
            results = self.graph.query(cypher_query, {"product": product_mentioned})
            
            if results:
                field_insights = [f"\n{r['rep']} visit with {r['hcp']} on {r['date']}:\n{r['insight']}" 
                                 for r in results]
                context = f"Field insights for {product_mentioned}:\n" + "\n".join(field_insights)
            else:
                context = f"No field insights found for {product_mentioned}"
            
        # Compare XAI vs Field insights
        elif "compare" in question.lower() and "xai" in question.lower() and "field" in question.lower():
            if product_mentioned:
                cypher_query = """
                    MATCH (p:Product {name: $product})
                    MATCH (v:Visit)-[:DISCUSSED]->(p)
                    MATCH (v)-[:GENERATED]->(xai:Insight {type: "XAI"})
                    MATCH (v)-[:RESULTED_IN]->(field:Insight {type: "Field"})
                    MATCH (field)-[r:RELATES_TO]->(xai)
                    MATCH (sr:SalesRep)-[:CONDUCTED]->(v)-[:WITH]->(hcp:HCP)
                    RETURN sr.name as rep, hcp.name as hcp, v.date as date,
                           xai.content as xai_insight, field.content as field_insight,
                           r.type as relationship_type
                    ORDER BY v.date DESC
                """
                
                results = self.graph.query(cypher_query, {"product": product_mentioned})
                
                if results:
                    comparisons = []
                    for r in results:
                        comparisons.append(f"\n{r['rep']} visit with {r['hcp']} on {r['date']}:")
                        comparisons.append(f"XAI Insight: {r['xai_insight']}")
                        comparisons.append(f"Field Insight: {r['field_insight']}")
                        comparisons.append(f"Relationship: {r['relationship_type']}")
                    
                    context = f"Comparison of XAI and Field insights for {product_mentioned}:\n" + "\n".join(comparisons)
                else:
                    context = f"No comparison data found for {product_mentioned}"
        
        # HCP-specific queries
        elif hcp_mentioned:
            cypher_query = """
                MATCH (hcp:HCP {name: $hcp})
                MATCH (v:Visit)-[:WITH]->(hcp)
                MATCH (v)-[:DISCUSSED]->(p:Product)
                MATCH (v)-[:GENERATED]->(xai:Insight {type: "XAI"})
                MATCH (v)-[:RESULTED_IN]->(field:Insight {type: "Field"})
                MATCH (sr:SalesRep)-[:CONDUCTED]->(v)
                RETURN sr.name as rep, p.name as product, v.date as date,
                       xai.content as xai_insight, field.content as field_insight
                ORDER BY v.date DESC
            """
            
            results = self.graph.query(cypher_query, {"hcp": hcp_mentioned})
            
            if results:
                hcp_visits = []
                for r in results:
                    hcp_visits.append(f"\nVisit on {r['date']} by {r['rep']} discussing {r['product']}:")
                    hcp_visits.append(f"XAI Insight: {r['xai_insight']}")
                    hcp_visits.append(f"Field Insight: {r['field_insight']}")
                
                context = f"Visits with {hcp_mentioned}:\n" + "\n".join(hcp_visits)
            else:
                context = f"No visit data found for {hcp_mentioned}"
        
        # Sales Rep specific queries
        elif rep_mentioned:
            cypher_query = """
                MATCH (sr:SalesRep)
                WHERE sr.name = $rep_name OR sr.id = $rep_name
                MATCH (sr)-[:CONDUCTED]->(v:Visit)
                MATCH (v)-[:WITH]->(hcp:HCP)
                MATCH (v)-[:DISCUSSED]->(p:Product)
                MATCH (v)-[:RESULTED_IN]->(field:Insight {type: "Field"})
                RETURN hcp.name as hcp, p.name as product, v.date as date,
                       field.content as field_insight
                ORDER BY v.date DESC
            """
            
            results = self.graph.query(cypher_query, {"rep_name": rep_mentioned})
            
            if results:
                rep_visits = []
                for r in results:
                    rep_visits.append(f"\nVisit on {r['date']} with {r['hcp']} discussing {r['product']}:")
                    rep_visits.append(f"Field Insight: {r['field_insight']}")
                
                context = f"Visits by {rep_mentioned}:\n" + "\n".join(rep_visits)
            else:
                context = f"No visit data found for {rep_mentioned}"
        
        # Use the LLM for the final answer
        if context:
            # Get answer from LLM using the retrieved context
            prompt = f"""
            Based on the following pharmaceutical data, please answer this question:
            
            Question: {question}
            
            Data:
            {context}
            
            Please provide a detailed answer based solely on the information provided above.
            """
            
            llm_response = self.llm.invoke(prompt)
            answer = llm_response.content
        else:
            # If no specialized context, use the general QA chain
            result = self.chain.invoke(question)
            answer = result["result"]
            context = str(result.get("intermediate_steps", "No steps available"))
        
        return {
            "answer": answer,
            "cypher_query": cypher_query,
            "context": context
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