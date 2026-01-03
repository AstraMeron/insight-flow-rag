# InsightFlow RAG: CrediTrust Consumer Complaint Analysis

An end-to-end Retrieval-Augmented Generation (RAG) pipeline designed to analyze and retrieve insights from consumer financial complaints across four key business pillars: Credit Cards, Savings Accounts, Money Transfers, and Personal Loans.

## 🚀 Project Progress (Interim Submission)

### Task 1: EDA & Preprocessing
* Cleaned and preprocessed 124,000+ consumer complaints.
* Handled missing values and removed non-narrative entries.
* Conducted exploratory data analysis identifying significant class imbalances.

### Task 2: Vector Store & Indexing
* **Stratified Sampling:** Created a balanced dataset of 15,000 records to ensure proportional representation of rare categories (e.g., Money Transfers).
* **Text Chunking:** Implemented `RecursiveCharacterTextSplitter` (Size: 600, Overlap: 50) for precise semantic retrieval.
* **Embeddings:** Utilized `all-MiniLM-L6-v2` for efficient, local vector generation.
* **Vector Store:** Persisted 16,000+ chunks into a **ChromaDB** database for fast retrieval.

## 🛠️ Setup Instructions
1. Clone the repository: `git clone https://github.com/AstraMeron/insight-flow-rag`
2. Create a virtual environment: `python -m venv venv`
3. Install dependencies: `pip install -r requirements.txt`