#!/bin/bash
# Check if .env exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    echo "Please create a .env file with your OpenAI API key and Neo4j credentials."
    exit 1
fi

# Run the setup script if requested
if [ "$1" == "--setup" ]; then
    echo "Setting up pharmaceutical database..."
    python setup_pharma_db.py
fi

# Run the Streamlit app
echo "Starting Pharma RAG application..."
streamlit run pharma_app.py 