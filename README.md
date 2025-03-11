
# RAG Chatbot System

A Retrieval-Augmented Generation chatbot that answers questions based on document content. Maintains conversation history and adapts to document knowledge domains.

## FEATURES

### Document Processing

-    Ingests 20+ file formats (PDFs, code, Office docs, etc.)

-    Splits content using recursive text splitting

-    Uses HuggingFace embeddings (default: all-MiniLM-L6-v2)

-    Stores vectors in Chroma DB

### AI Chat Interface

-   Retrieves relevant document chunks

-   Uses OpenAI/Chat models for responses

-   Maintains conversation history

-   Features automatic domain analysis

-   Includes interaction logging

## INSTALLATION

1.  Clone repository:  
    git clone [https://github.com/Jammer23rd/rag-chatbot](https://github.com/Jammer23rd/rag-chatbot)  
    cd rag-chatbot
    
2.  Create virtual environment:  
    python -m venv .venv  
    source .venv/bin/activate # Linux/MacOS
    
     \# ..venv\Scripts\activate # Windows
    
3.  Install dependencies:  
    pip install -r requirements.txt
    

## USAGE

1.  Add documents to documents/ folder
    
2.  Process documents:  
    python load_and_split_docs.py
    
3.  Start chat interface:  
    python talk.py
    
4.  Optional arguments for talk.py:  
    --cpu Force CPU-only mode (useful for GPU compatibility issues)
    

## CONFIGURATION

Create .env file with these variables:

OPENAI_API_BASE="https://api.openai_compatible.com"  
OPENAI_API_KEY="your-api-key"  
MODEL_NAME="model_name"  
EMBEDDING_MODEL="all-MiniLM-L6-v2" [info](https://docs.trychroma.com/docs/embeddings/embedding-functions)  
CHROMA_PERSIST_DIR="db"  
HISTORY_LENGTH=5  
API_TIMEOUT=30

## CUSTOMIZATION

-   Modify documents/ with your files
    
-   Adjust chunk sizes in load_and_split_docs.py
    
-   Customize prompts in talk.py
    
-   Change logging format in ChatLogger class
    

## FOLDER STRUCTURE

.  
├── documents/ # User documents  
├── db/ # Vector database  
├── logs/ # Conversation logs  
├── .env # Configuration  
├── requirements.txt  
└── ... # Source files

## MAINTENANCE ⚠️  

### Warning: This will permanently delete:

All documents in documents/
Vector database in db/
Chat logs in logs/
Any processing artifacts### System Reset

### Run the reset

    chmod +x reset_system.sh
    ./reset_system.sh

## SECURITY NOTE

Rotate API keys if they appear in any logs or history files
