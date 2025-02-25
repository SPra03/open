import streamlit as st
from graph_rag import WikipediaGraphRAG
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the RAG system
@st.cache_resource
def init_rag_system():
    return WikipediaGraphRAG()

def main():
    st.title("Simple RAG App")
    
    # Sidebar for indexing
    with st.sidebar:
        st.header("Indexes")
        
        # Wikipedia page input
        wiki_page = st.text_input("Enter a Wikipedia page title to index")
        
        # Options
        split_docs = st.checkbox("Split documents", value=True)
        overwrite = st.checkbox("Overwrite index", value=True)
        
        # Max document length slider
        max_length = st.number_input("Max document length", 
                                   min_value=100, 
                                   max_value=1000, 
                                   value=180)
        
        # Add Index button
        if st.button("Add Index"):
            if wiki_page:
                rag = init_rag_system()
                with st.spinner(f"Indexing {wiki_page}..."):
                    try:
                        rag.fetch_wikipedia_content([wiki_page])
                        st.success(f"Successfully indexed {wiki_page}")
                    except Exception as e:
                        st.error(f"Error indexing {wiki_page}: {str(e)}")
    
    # Main content area
    rag = init_rag_system()
    topics = rag.get_stored_topics()
    
    st.subheader("Select an Index")
    st.caption("Choose a topic to query about from your knowledge base")
    selected_index = st.selectbox("", topics if topics else ["No indexes available"])
    
    # Show information about selected index
    if selected_index in topics:
        with st.expander("Selected Topic Information"):
            topic_info = rag.graph.query("""
                MATCH (t:Topic {name: $topic})
                RETURN t.summary as summary
            """, {"topic": selected_index})
            if topic_info:
                st.write("Summary:", topic_info[0]["summary"])
    
    # Question input
    st.subheader("Enter a question")
    st.caption(f"Ask a question about {selected_index}")
    question = st.text_input("")
    
    # Submit button
    if st.button("Submit"):
        if question:
            with st.spinner("Generating answer..."):
                try:
                    result = rag.query(question, selected_index)
                    
                    # Display results
                    st.subheader("Answer")
                    st.write(result["answer"])
                    
                    # Display context in expander
                    with st.expander("Show Context"):
                        st.text("Cypher Query:")
                        st.code(result["cypher_query"], language="cypher")
                        st.text("Retrieved Context:")
                        st.write(result["context"])
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
        else:
            st.warning("Please enter a question")

if __name__ == "__main__":
    main() 