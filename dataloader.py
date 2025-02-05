# data_loader.py
import os
import glob
from sentence_transformers import SentenceTransformer

def load_documents_from_folder(folder_path="docs"):
    """
    Reads all .txt files from the specified folder and returns a list of document contents.
    """
    documents = []
    file_paths = glob.glob(os.path.join(folder_path, "*.txt"))
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:  # Only add non-empty documents
                documents.append(content)
    return documents

def compute_document_embeddings(documents, model_name='all-MiniLM-L6-v2'):
    """
    Compute and return embeddings for each document.
    """
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(documents, convert_to_tensor=False)
    return embeddings, embedder

if __name__ == "__main__":
    # For quick testing of document loading:
    docs = load_documents_from_folder("docs")
    print("Loaded Documents:")
    for idx, doc in enumerate(docs, 1):
        print(f"{idx}: {doc[:100]}...")  # print first 100 characters of each document
