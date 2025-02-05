# data_loader.py
import os
import glob
from sentence_transformers import SentenceTransformer
from docx import Document 
import pandas as pd
import numpy as np
import fitz

def load_documents_from_folder(folder_path=""):
    """
    Reads all .txt files from the specified folder and returns a list of document contents.
    """
    documents = {}
    file_paths = glob.glob(os.path.join(folder_path, "*.pdf"))
  #  print("file paths are",file_paths)
    for file_path in file_paths:
        try: 
            doc = fitz.open(file_path)
   #         print("first check")
            content = "\n".join([page.get_text("text") for page in doc])  # Extract text from paragraphs
            doc.close()

            #documents.append(content)
    #        print("Sumedha checks here")
    #        print(content)
    #        print("os is :", os.path.basename(file_path))
            documents[os.path.basename(file_path)] = content  
        except Exception as e: 
            print(f"Skipping file {file_path} due to error: {e}")
        #with open(file_path, "r", encoding="utf-8") as f:
        #    content = f.read().strip()
        #    if content:  # Only add non-empty documents
        #        documents.append(content)
    return documents

def compute_document_embeddings(documents, model_name='all-MiniLM-L6-v2'):
    """
    Compute and return embeddings for each document.
    """
    embedder = SentenceTransformer(model_name)
    embeddings = {name: np.array(embedder.encode(content, convert_to_tensor=False))
                  for name, content in documents.items()}
    return embeddings

def save_embeddings_to_csv(embeddings, folder="embeddings"):
    """
    Save embeddings to a CSV file.
    """
    #print("here i am")
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, "document_embeddings.csv")
    #print(file_path)
    #df = pd.DataFrame(embeddings).T  # Transpose so filenames are rows
    df = pd.DataFrame.from_dict(embeddings, orient="index") 
    df.to_csv(file_path, index_label="Document")
    print(f"Embeddings saved to {file_path}")

if __name__ == "__main__":
    # For quick testing of document loading:
    docs = load_documents_from_folder("")
    print("Loaded Documents:")
    #print(docs)
    for idx, doc in enumerate(docs, 1):
        print(f"{idx}: {doc[:100]}...")  # print first 100 characters of each document
    
    if docs:
        embeddings = compute_document_embeddings(docs)
    #    print(embeddings)
        save_embeddings_to_csv(embeddings)
    else:
        print("No documents found for embedding computation.")
