import streamlit as st
import pandas as pd
from pharma_rag import PharmaGraphRAG
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the RAG system
@st.cache_resource
def init_pharma_rag_system():
    return PharmaGraphRAG()

def main():
    st.title("Pharmaceutical Insights RAG")
    
    # Sidebar for data loading
    with st.sidebar:
        st.header("Data Management")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload pharmaceutical data (CSV)", type=["csv"])
        
        # Process data button
        if uploaded_file is not None:
            if st.button("Process Data"):
                rag = init_pharma_rag_system()
                with st.spinner("Processing pharmaceutical data..."):
                    try:
                        # Save the uploaded file temporarily
                        with open("temp_data.csv", "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Clear existing data if needed
                        if st.checkbox("Clear existing data"):
                            rag.clear_database()
                        
                        # Process the data
                        rag.process_pharma_data("temp_data.csv")
                        st.success("Successfully processed pharmaceutical data")
                    except Exception as e:
                        st.error(f"Error processing data: {str(e)}")
        
        # Database stats
        st.subheader("Database Statistics")
        rag = init_pharma_rag_system()
        
        try:
            products = rag.get_products()
            hcps = rag.get_hcps()
            reps = rag.get_reps()
            
            st.write(f"Products: {len(products)}")
            st.write(f"Healthcare Providers: {len(hcps)}")
            st.write(f"Sales Representatives: {len(reps)}")
        except:
            st.write("Database not yet initialized")
    
    # Main content - Tabs
    tab1, tab2, tab3 = st.tabs(["Query Insights", "Browse Data", "Analytics"])
    
    with tab1:
        st.header("Query Pharmaceutical Insights")
        
        # Query input
        query = st.text_area("Enter your question", height=100, 
                            placeholder="e.g., What do field reps say about Cardiofix?")
        
        # Submit button
        if st.button("Submit Query"):
            if query:
                rag = init_pharma_rag_system()
                with st.spinner("Generating insights..."):
                    try:
                        result = rag.query(query)
                        
                        # Display the answer
                        st.subheader("Answer")
                        st.write(result["answer"])
                        
                        # Show the supporting context
                        with st.expander("Show Details"):
                            st.subheader("Context Information")
                            st.write(result["context"])
                            
                            if result["cypher_query"]:
                                st.subheader("Database Query")
                                st.code(result["cypher_query"], language="cypher")
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
            else:
                st.warning("Please enter a question")
    
    with tab2:
        st.header("Browse Pharmaceutical Data")
        
        rag = init_pharma_rag_system()
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            product_filter = st.selectbox("Filter by Product", ["All Products"] + rag.get_products())
        
        with col2:
            hcp_filter = st.selectbox("Filter by HCP", ["All HCPs"] + rag.get_hcps())
        
        with col3:
            rep_filter = st.selectbox("Filter by Sales Rep", ["All Reps"] + rag.get_reps())
        
        # Build query based on filters
        query_params = {}
        where_clauses = []
        
        if product_filter != "All Products":
            where_clauses.append("p.name = $product")
            query_params["product"] = product_filter
        
        if hcp_filter != "All HCPs":
            where_clauses.append("hcp.name = $hcp")
            query_params["hcp"] = hcp_filter
        
        if rep_filter != "All Reps":
            where_clauses.append("sr.name = $rep")
            query_params["rep"] = rep_filter
        
        where_clause = " AND ".join(where_clauses)
        if where_clause:
            where_clause = "WHERE " + where_clause
        
        # Query for visit data
        cypher_query = f"""
            MATCH (sr:SalesRep)-[:CONDUCTED]->(v:Visit)-[:WITH]->(hcp:HCP)
            MATCH (v)-[:DISCUSSED]->(p:Product)
            MATCH (v)-[:GENERATED]->(xai:Insight {{type: "XAI"}})
            MATCH (v)-[:RESULTED_IN]->(field:Insight {{type: "Field"}})
            {where_clause}
            RETURN sr.name as Rep, hcp.name as HCP, p.name as Product, 
                   v.date as Date, xai.content as "XAI Insight", 
                   field.content as "Field Insight"
            ORDER BY v.date DESC
        """
        
        try:
            results = rag.graph.query(cypher_query, query_params)
            if results:
                df = pd.DataFrame(results)
                st.dataframe(df)
            else:
                st.info("No data found with the selected filters")
        except Exception as e:
            st.error(f"Error querying data: {str(e)}")
    
    with tab3:
        st.header("Pharmaceutical Insights Analytics")
        
        rag = init_pharma_rag_system()
        
        # Insight comparison analysis
        st.subheader("Insight Relationship Analysis")
        try:
            relationship_stats = rag.graph.query("""
                MATCH (field:Insight {type: "Field"})-[r:RELATES_TO]->(xai:Insight {type: "XAI"})
                RETURN r.type as relationship, count(*) as count
            """)
            
            if relationship_stats:
                # Convert to dataframe for charting
                rel_df = pd.DataFrame(relationship_stats)
                st.bar_chart(rel_df.set_index('relationship'))
            else:
                st.info("No relationship data available")
                
            # Product insights analysis
            st.subheader("Insights by Product")
            product_stats = rag.graph.query("""
                MATCH (p:Product)<-[:DISCUSSED]-(v:Visit)
                MATCH (v)-[:RESULTED_IN]->(field:Insight {type: "Field"})
                RETURN p.name as product, count(field) as field_insights
                ORDER BY field_insights DESC
            """)
            
            if product_stats:
                prod_df = pd.DataFrame(product_stats)
                st.bar_chart(prod_df.set_index('product'))
            else:
                st.info("No product insight data available")
                
        except Exception as e:
            st.error(f"Error generating analytics: {str(e)}")

if __name__ == "__main__":
    main() 