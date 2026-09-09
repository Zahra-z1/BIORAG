# 🧬 BioRAG

BioRAG is a Retrieval-Augmented Generation (RAG) application designed to answer biology-related questions using information retrieved from indexed PDF documents.

The system combines lexical retrieval, semantic search, document-aware reranking, and grounded answer extraction to provide answers with source references and page numbers.

## 🌐 Live Demo

**Streamlit App:**
https://zahra-z1-biorag-app-lswtdv.streamlit.app/

---

## ✨ Features

* Biology-focused RAG system
* PDF document ingestion
* Text chunking with overlap
* TF-IDF lexical retrieval
* Semantic retrieval using Sentence Transformers
* FAISS vector search
* Hybrid relevance scoring
* Document-title-aware reranking
* Definition-aware retrieval
* Source and page citations
* Retrieval confidence scores
* Extractive grounded answers
* Optional Transformer-based generation
* Streamlit frontend
* FastAPI backend
* Local and cloud deployment support

---

## 🧠 Example Questions

You can ask questions such as:

* What is bioinformatics?
* Define genome.
* What is a gene?
* What is transcription?
* What is the cell cycle?
* Explain DNA replication.
* Explain the structure of proteins.

---

## 🏗️ Project Structure

```text
BIORAG/
│
├── app.py
├── api.py
├── requirements.txt
├── .env.example
├── .gitignore
│
├── src/
│   ├── config.py
│   ├── retrieval.py
│   ├── generation.py
│   ├── embeddings.py
│   ├── prompts.py
│   └── ...
│
├── scripts/
│   └── build_index.py
│
├── data/
│   └── ...
│
├── vectorstore/
│   ├── metadata.json
│   └── ...
│
├── setup_windows.ps1
├── run_frontend_windows.ps1
└── run_api_windows.ps1
```

---

## 🔎 How It Works

BioRAG follows a retrieval-first pipeline:

```text
User Question
      ↓
Question Analysis
      ↓
Lexical Retrieval
      +
Semantic Retrieval
      ↓
Candidate Reranking
      ↓
Relevant PDF Chunks
      ↓
Grounded Answer Extraction
      ↓
Answer + Source + Page + Relevance
```

### 1. Document Retrieval

The system searches indexed PDF chunks using:

* TF-IDF lexical similarity
* Sentence Transformer embeddings
* FAISS semantic similarity

### 2. Reranking

Retrieved chunks are reranked using signals such as:

* Query relevance
* Semantic similarity
* Subject coverage
* Definition answerability
* Document title relevance
* Content quality
* Nearby section context

### 3. Answer Generation

The application currently supports:

* `extractive`
* `transformers`
* `auto`

The extractive mode keeps answers grounded directly in the indexed source material.

---

## ⚙️ Environment Variables

Create a `.env` file in the project root.

Example:

```env
# Retrieval
TOP_K=6
CHUNK_SIZE=1200
CHUNK_OVERLAP=220
RELEVANCE_THRESHOLD=0.22
ENABLE_SEMANTIC=true
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Answer generation
GENERATION_BACKEND=extractive
GENERATION_MODEL=google/flan-t5-base
MAX_NEW_TOKENS=420
MAX_CONTEXT_CHARS=12000

# Allow Hugging Face model downloads
ALLOW_MODEL_DOWNLOAD=true
```

Do not commit your real `.env` file to GitHub.

Use `.env.example` for public configuration examples.

---

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Zahra-z1/BIORAG.git
cd BIORAG
```

### 2. Create a virtual environment

Using Python 3.11 is recommended.

```powershell
py -3.11 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy:

```text
.env.example
```

to:

```text
.env
```

and update the values if required.

### 5. Build the knowledge index

If the vector index has not already been generated:

```bash
python -m scripts.build_index
```

### 6. Run the Streamlit frontend

```bash
streamlit run app.py --server.fileWatcherType none
```

Open:

```text
http://localhost:8501
```

---

## 🔌 Run the FastAPI Backend

The project also includes a FastAPI backend.

Run:

```bash
uvicorn api:app --reload
```

The API will normally be available at:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 📚 Knowledge Base

BioRAG uses biology-related PDF documents covering topics such as:

* Bioinformatics
* Genetics
* Genomics
* DNA
* Genes
* Transcription
* Translation
* Protein structure
* Molecular biology

The application retrieves relevant passages from these documents and displays the source PDF and page number with the answer.

---

## 🛠️ Technologies

* Python
* Streamlit
* FastAPI
* Scikit-learn
* Sentence Transformers
* FAISS
* Hugging Face Transformers
* PyTorch
* NumPy
* PDF processing tools

---

## ☁️ Deployment

The Streamlit application is deployed using Streamlit Community Cloud.

Live version:

https://zahra-z1-biorag-app-lswtdv.streamlit.app/

Updates pushed to the GitHub `main` branch can be automatically reflected in the deployed application.

---

## 🔐 Security

Sensitive configuration should never be committed to the repository.

The following files/directories should remain ignored:

```text
.env
.venv/
__pycache__/
.streamlit/secrets.toml
```

For cloud deployment, environment variables should be configured using the hosting platform's secret-management system.

---

## 📌 Current Answer Mode

The current recommended configuration is:

```env
GENERATION_BACKEND=extractive
```

This prioritizes grounded answers taken directly from retrieved PDF evidence.

---

## 📄 License

No license has currently been specified for this project.
