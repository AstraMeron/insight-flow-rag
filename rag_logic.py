import chromadb
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# 1. LOAD SECRETS
# This looks for the .env file in the same directory
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("⚠️ HF_TOKEN not found! Check your .env file.")

client = InferenceClient(api_key=HF_TOKEN)

# 2. FIND THE DATABASE DYNAMICALLY
current_file_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_file_dir, "notebooks", "chroma_db")

print(f"Connecting to database at: {db_path}")
client_db = chromadb.PersistentClient(path=db_path)

# 3. LOAD COLLECTION
collection = client_db.get_or_create_collection(name="complaints_collection")

def get_relevant_context(query, k=5):
    """Retrieves top chunks from the vector store."""
    results = collection.query(
        query_texts=[query],
        n_results=k
    )
    # Extract data safely
    docs = results.get('documents', [[]])[0]
    metas = results.get('metadatas', [[]])[0]
    context = "\n".join(docs)
    return context, docs, metas

def generate_rag_response(query, context):
    prompt = f"""You are a financial analyst assistant for CrediTrust. 
Answer the question based ONLY on the provided context. 
If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question: 
{query}

Answer:"""

    completion = client.chat.completions.create(
        model="HuggingFaceH4/zephyr-7b-beta",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.7
    )
    return completion.choices[0].message.content