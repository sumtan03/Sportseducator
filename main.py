import os
from document_loader import load_documents_from_folder
from document_loader import compute_document_embeddings
from document_loader import save_embeddings_to_csv
from rag import RAG
from sentence_transformers import SentenceTransformer

OPENAI_API_KEY = ""  
#DOCUMENTS_PATH = "documents"  # Folder containing .docx files

def main():
    # Step 1: Load and embed documents
    #loader = DocumentLoader()
    documents = load_documents_from_folder()
    print("hey there")
    if documents:
        print(f"Loaded {len(documents)} documents.")
        embeddings = compute_document_embeddings(documents)
        ### saving embeddings 
        print("this is me, there is no where else i would rather be")
        save_embeddings_to_csv(embeddings)

    # Step 2: Initialize RAG system
    model = SentenceTransformer("all-MiniLM-L6-v2")
    rag = RAG(openai_api_key=OPENAI_API_KEY)

    # Step 3: Ask a query
    while True:
        query = input("\nEnter your question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        response = rag.ask_question(query, k=3)
        print("\nAnswer:", response)

if __name__ == "__main__":
    openai_api_key=""
    main()
