import openai
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import os
from docx import Document
import fitz
import re

class RAG:
    def __init__(self, openai_api_key, embeddings_folder="embeddings"):
        self.openai_api_key = openai_api_key
        self.embeddings_folder = embeddings_folder
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.docs_folder="C:/Users/staneja/Cap-test/Assignment/docs"

    def sliding_window_chunk(self,text, chunk_size=1000, overlap=200):
        words = text.split()
        chunks = []
    
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            print(chunk)
        print("\n chunks are:", chunks)
        return chunks


    def load_document_content(self, file_path):
        """ given a file path, load the document"""
        print("we are in load doc")
        print("file path:", file_path)
        ext=os.path.splitext(file_path)[-1].lower()
        if ext == ".pdf":
            try:
                doc=fitz.open(file_path)
                text=[]
                for page in doc:
                    text.append(page.get_text("text"))
             #   content = "\n".join([page.get_text("text") for page in doc])  # Extract text from paragraphs
                content = "\n".join(text)
                # Remove extra spaces and line breaks
                content = " ".join(content.split())
                # Ensure chunks are meaningful
                if len(content.split()) < 20:  # If text is too short, document extraction failed
                    print("Warning: Extracted document content is too short!")
                    return []
                print("Original Document Length:", len(content.split()), "words")
                 # Apply overlapping chunking
                chunks = self.sliding_window_chunk(content, chunk_size=1000, overlap=200)
                print(f"Generated {len(chunks)} chunks from document")
            
                return chunks  # Return a list of chunks instead of a single string
                #print(content)
                #return content.strip()
            except Exception as e:
                print(f"Error processing file {file_path}:{e}")
                return ""
        else:
            print("unsupported file extention")
            return ""

    def load_embeddings(self):
        """
        Load embeddings from CSV.
        """
        file_path = os.path.join(self.embeddings_folder, "document_embeddings.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError("Embeddings file not found!")

        df = pd.read_csv(file_path, index_col="Document")
        doc_names = list(df.index)
        embeddings = df.to_numpy()
        return embeddings, doc_names

    
    def clean_text(self,text):
        """Clean extracted text by fixing encodings and removing junk characters."""
        text = text.encode("utf-8", "ignore").decode("utf-8")  # Fix encoding
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
        text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters (if unwanted)
        return text.strip()


    def retrieve_relevant_docs(self, query, k=3):
        """.
        Retrieve top-k relevant documents using cosine similarity.
        """
    #    print(" \n\n am in retrieve_relevant_docs")
        embeddings, doc_names = self.load_embeddings()
    #    print("\n\n doc name is :",doc_names)
    #    print(embeddings)
        # Encode query to embedding
        query_embedding = np.array(self.model.encode(query, convert_to_tensor=False))
    #    print("query_embedding is :",query_embedding)
        # Compute cosine similarity between query and all document embeddings
        similarities = [1 - cosine(query_embedding, emb) for emb in embeddings]

        # Get indices of top-k most similar docs
        #top_k_indices = np.argsort(similarities)[-k:][::-1]
        similarities = np.array(similarities)
        top_k_indices = similarities.argsort()[-k:][::-1] 
        relevant_file = [doc_names[i] for i in top_k_indices]
    #    print("relevant file names are :")
        for file in relevant_file:
            print(file[:200])
        
        relevant_docs = []
        for file_name in relevant_file:
        # Construct full file path by joining docs_folder and file_name
            file_path = os.path.join(self.docs_folder, file_name)
            print(file_path)
            doc_chunks = self.load_document_content(file_path)
            if isinstance(doc_chunks, list):  # Flattening the chunks into relevant_docs
                relevant_docs.extend(doc_chunks)
                print("\n relevant docs are :", relevant_docs)
            else:
                relevant_docs.append(doc_chunks)
            #relevant_docs.append(content)
            
        return relevant_docs

    def query_openai(self, query, retrieved_docs):
        """
        Query OpenAI API with the retrieved documents as context.
        """
        few_shot_examples = (
                                "Example 1:\n"
                                "Context: 'Cricket World Cup 2023 was held in India. India won the tournament.'\n"
                                "Question: 'Who won the Cricket World Cup 2023?'\n"
                                "Answer: 'India.'\n\n"
                                "Example 2:\n"
                                "Context: 'The US Open 2024 was won by Novak Djokovic.'\n"
                                "Question: 'Who won the US Open 2024?'\n"
                                "Answer: 'Novak Djokovic.'\n\n"   
                              #  "Example 3:\n"
                              #  "Context: ''\n"
                              #  "Question: 'What are you about?'\n"
                              #  "Answer: 'US Open 2024, FIFA World Cup 2022 and Cricket World Cup 2023.'\n\n"
                              #  "Example 4:\n"
                              #  "Context: ''\n"
                              #  "Question: 'What is your reference?'\n"
                              #  "Answer: 'US Open 2024, FIFA World Cup 2022 and Cricket World Cup 2023.'\n\n"
                )
         # Limit the number of chunks to fit within context size (approx. 10 chunks)
        # Ensure retrieved documents contain useful text
        if not retrieved_docs:
            print("No retrieved docs, returning default response.")
            return "I Cannot respond"
        context = "\n".join(retrieved_docs[:3])
        context=self.clean_text(context)
        user_query = "what do you know"
        print("\n\n\n\n open ai")
        print(context)
        print("\n query is:")
        print(query)
       # prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"
        client = openai.OpenAI(api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", 
                       "content": ("You are an AI assistant that answers questions using the provided context. "
                       "If the context does not contain enough information, try to infer. "
                       "You should base your identity and responses solely on the documents provided. "
                        "Otherwise, say: 'I don't know'."
                        "Provide structured responses if applicable."
                        "You are a sports guidance assistant"
                        "Do not state you are an AI assistant unless explicitly asked"
                                )
                    },
                    {
                        "role": "user",
                        "content": f"{few_shot_examples}\n\nContext:\n{context}\n\nQuestion:\n{query}"
                        #"content": f"Context:\n{context}\n\nQuestion:\n{query}"
                    }
                ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content

    def ask_question(self, query, k=3):
        """
        End-to-end retrieval + OpenAI response.
        """
        relevant_docs = self.retrieve_relevant_docs(query, k)
        answer = self.query_openai(query, relevant_docs)
        return answer
