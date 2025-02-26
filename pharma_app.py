import streamlit as st
import pandas as pd
from pharma_rag import PharmaGraphRAG
import os
from dotenv import load_dotenv
import plotly.express as px

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
        
        # Show available products
        rag = init_pharma_rag_system()
        products = rag.get_products()
        
        with st.expander("Available Products"):
            product_list = ", ".join([p.split(" (")[0] for p in products])
            st.write(f"You can ask about: {product_list}")
        
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
            product_filter = st.selectbox(
                "Filter by Product", 
                ["All Products"] + rag.get_products(),
                key="browse_product_filter"
            )
        
        with col2:
            hcp_filter = st.selectbox(
                "Filter by HCP", 
                ["All HCPs"] + rag.get_hcps(),
                key="browse_hcp_filter"
            )
        
        with col3:
            rep_filter = st.selectbox(
                "Filter by Sales Rep", 
                ["All Reps"] + rag.get_reps(),
                key="browse_rep_filter"
            )
        
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
            MATCH (sr:SalesRep)
            MATCH (v:Visit)
            MATCH (hcp:HCP)
            MATCH (p:Product)
            MATCH (xai:Insight {{type: "XAI"}})
            MATCH (field:Insight {{type: "Field"}})
            WHERE (sr)-[:CONDUCTED]->(v)
            AND (v)-[:WITH]->(hcp)
            AND (v)-[:DISCUSSED]->(p)
            AND (v)-[:GENERATED]->(xai)
            AND (v)-[:RESULTED_IN]->(field)
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
        
        # Check if data exists in the database
        data_exists = rag.graph.query("""
            MATCH (n) RETURN count(n) as count
        """)
        
        if data_exists and data_exists[0]['count'] == 0:
            st.warning("No data found in the database. Please upload and process data first.")
        else:
            # Insight comparison analysis
            st.subheader("XAI vs Field Insight Analysis")
            
            # Product selector for filtering
            product_filter = st.selectbox(
                "Filter by Product",
                ["All Products"] + rag.get_products(),
                key="analytics_product_filter"
            )
            
            product_name = None if product_filter == "All Products" else product_filter
            
            try:
                insight_analysis = rag.analyze_insight_relationships(product_name)
                
                # Display relationship statistics
                st.write("### Relationship Distribution")
                if insight_analysis["stats"]:
                    stats_df = pd.DataFrame(insight_analysis["stats"])
                    fig = px.pie(stats_df, values='count', names='relationship', 
                                 title='XAI vs Field Insight Relationships')
                    st.plotly_chart(fig)
                else:
                    st.info("No relationship data available")
                
                # Display top contradictions
                st.write("### Top Contradictions")
                if insight_analysis["contradictions"]:
                    for i, contradiction in enumerate(insight_analysis["contradictions"]):
                        with st.expander(f"{i+1}. {contradiction['product']} - Confidence: {contradiction['confidence']:.2f}"):
                            st.write("**XAI Insight:**")
                            st.write(contradiction['xai_insight'])
                            st.write("**Field Insight:**")
                            st.write(contradiction['field_insight'])
                            st.write("**Analysis:**")
                            st.write(contradiction['summary'])
                
            except Exception as e:
                st.error(f"Error analyzing insights: {str(e)}")

if __name__ == "__main__":
    main() 