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
* Retrieval relevance scores
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
* Explain DNA replication.
* Explain the structure of proteins.
* What is DNA?

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
├── README.md
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
 ┌───────────────┐
 │               │
 ▼               ▼
Lexical       Semantic
Retrieval     Retrieval
(TF-IDF)      (Embeddings)
 │               │
 └───────┬───────┘
         ↓
 Hybrid Relevance Scoring
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

The `extractive` mode keeps answers grounded directly in the indexed source material.

---

## ⚙️ Environment Variables

Create a `.env` file in the project root.

Example configuration:

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

**Do not commit your real `.env` file to GitHub.**

Use `.env.example` for public configuration examples.

---

# 🚀 Run Locally

## 1. Clone the Repository

```bash
git clone https://github.com/Zahra-z1/BIORAG.git
cd BIORAG
```

## 2. Create a Virtual Environment

Python 3.11 is recommended.

```powershell
py -3.11 -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## 4. Configure Environment Variables

Copy `.env.example` to `.env` and update the values if required.

```text
.env.example → .env
```

Do not commit the real `.env` file to GitHub.

## 5. Build the Knowledge Index

If the vector index has not already been generated:

```bash
python -m scripts.build_index
```

This prepares the searchable knowledge base from the PDF documents.

## 6. Run the Streamlit Frontend

```bash
streamlit run app.py --server.fileWatcherType none
```

Open the application at:

```text
http://localhost:8501
```

---

# 🔌 Run the FastAPI Backend

The project also includes a FastAPI backend.

Start the backend with:

```bash
uvicorn api:app --reload
```

The API will normally be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

---

# 📚 Knowledge Base

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

# 🛠️ Technologies

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

# ☁️ Deployment

The Streamlit application is deployed using **Streamlit Community Cloud**.

**Live application:**
https://zahra-z1-biorag-app-lswtdv.streamlit.app/

The deployed application is connected to the GitHub repository. Changes pushed to the configured GitHub branch can trigger an automatic redeployment through Streamlit Community Cloud.

---

# 🔐 Security

Sensitive configuration should never be committed to the repository.

The following files and directories should remain ignored:

```text
.env
.venv/
__pycache__/
.streamlit/secrets.toml
```

For cloud deployment, environment variables and secrets should be configured using the hosting platform's secret-management system.

---

# 📌 Current Answer Mode

The current recommended configuration is:

```env
GENERATION_BACKEND=extractive
```

This prioritizes grounded answers taken directly from retrieved PDF evidence.

---

# 🎯 Project Goal

BioRAG was developed to demonstrate the application of Retrieval-Augmented Generation to biology and bioinformatics research assistance.

The project focuses on:

* Domain-specific information retrieval
* Combining lexical and semantic search
* Improving retrieval through reranking
* Grounding answers in source documents
* Providing transparent source and page references
* Building a practical research-assistance interface

---

# 🔮 Future Improvements

Potential future improvements include:

* Larger biology and biomedical knowledge bases
* More specialized biomedical embedding models
* Improved hybrid retrieval strategies
* Automated retrieval and answer-quality evaluation
* Advanced reranking models
* Support for additional document formats
* Improved conversational memory
* Faster inference and indexing
* Docker-based deployment
* Scalable cloud deployment
* Automated evaluation benchmarks

---

# ⚠️ Limitations

BioRAG's answers depend on the documents available in its knowledge base.

If relevant information is not present in the indexed documents, the system may not be able to provide an appropriate answer.

BioRAG should therefore be treated as a **research-assistance and educational tool**, rather than a replacement for scientific literature, textbooks, or expert review.

---

# 📄 License

No license has currently been specified for this project.

If you plan to distribute or allow others to reuse the project, consider adding an appropriate open-source license such as MIT, Apache-2.0, or GPL-3.0.

---

## 👩‍💻 Author

**Zahra**

Bioinformatics Student

Interested in Bioinformatics, Machine Learning, RAG Systems, and AI-powered research tools.
