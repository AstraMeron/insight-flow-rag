import pandas as pd
import chromadb
import os

# 1. SETUP PATHS
root_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(root_dir, "notebooks", "chroma_db")
csv_path = os.path.join(root_dir, "data", "processed", "filtered_complaints.csv")

# 2. LOAD DATA
print("Reading processed data...")
df = pd.read_csv(csv_path).head(100)

# --- NEW: DEBUGGING COLUMN NAMES ---
print(f"Available columns: {df.columns.tolist()}")

# Automatically pick the correct column names
# We look for common names if 'complaint_text' isn't there
target_col = None
for col in ['complaint_text', 'Consumer complaint narrative', 'complaint', 'text']:
    if col in df.columns:
        target_col = col
        break

if not target_col:
    raise KeyError(f"Could not find a text column! Your columns are: {df.columns.tolist()}")

product_col = 'Product' if 'Product' in df.columns else df.columns[0] # Fallback to first col
# -----------------------------------

# 3. INITIALIZE CHROMA
client = chromadb.PersistentClient(path=db_path)
collection = client.get_or_create_collection(name="complaints_collection")

# 4. INGEST
print(f"Using column '{target_col}' for ingestion...")
collection.add(
    documents=df[target_col].astype(str).tolist(), # Force to string to avoid errors
    ids=[str(i) for i in range(len(df))],
    metadatas=[{"product": str(p)} for p in df[product_col].tolist()]
)
print(f"✅ Ingestion Complete! Added {len(df)} records.")