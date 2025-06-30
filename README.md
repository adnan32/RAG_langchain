# Local RAG Q&A Demo

A Streamlit-based front-end for experimenting with Retrieval-Augmented Generation (RAG) using LangChain.  
Upload a PDF or text file, configure your pipeline (splitting, embeddings, vector store, retriever, chain type), then ask questions against your document.

---

## 🔧 Features

- **File Upload & Preprocessing**  
  Extracts and cleans text from PDFs or plain text files.

- **Configurable Text Splitting**  
  Choose from paragraph-, token-, sentence- or header-based chunkers.

- **Pluggable Embeddings**  
  Lightweight to high-precision models from HuggingFace.

- **Multiple Vector Stores**  
  Local Chroma or FAISS backends.

- **Advanced Retriever Techniques**  
  Basic vector search, multi-query expansion, or contextual compression.

- **Flexible Chain Types**  
  `stuff`, `map_reduce`, `refine`, `map_rerank` — trade off speed, accuracy, and context handling.

- **Query Rewriting**  
  Automatically refocuses your question for more effective retrieval.

---

## 🚀 Getting Started

### 1. Clone and install

```bash
git clone https://github.com/yourusername/local-rag-demo.git
cd local-rag-demo
pip install -r requirements.txt
```

> **Requirements**  
> - Python 3.8+  
> - Streamlit  
> - LangChain + dependencies (Ollama LLM, HuggingFace embeddings, Chroma or FAISS)

### 2. Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` and follow the on-screen instructions.

---

## ⚙️ Configuration Options

Before processing your file, you can customize how your document is chunked, embedded, stored, retrieved, and used by the language model.

### 🔹 Split Strategy

| Splitter | Description |
|----------|-------------|
| `RecursiveCharacterTextSplitter` | Smart paragraph/sentence splits (`chunk_size=1000`, `overlap=200`) |
| `TokenTextSplitter` | Exact control by model token count |
| `NLTKTextSplitter` | Sentence detection using NLTK |
| `MarkdownHeaderTextSplitter` | Splits at `#`, `##`, `###` headers |

### 🔹 Embedding Model

| Model | Description |
|-------|-------------|
| `all-MiniLM-L6-v2` | Fast & lightweight |
| `BAAI/bge-base-en-v1.5` | Balanced accuracy for QA tasks |
| `intfloat/e5-large` | High-quality embeddings but slower |

### 🔹 Vector Store

| Store | Description |
|-------|-------------|
| `Chroma` | Simple local DB with metadata, persistent across sessions |
| `FAISS` | In-memory, very fast, ideal for prototyping |

### 🔹 Retriever Technique

| Technique | Description |
|-----------|-------------|
| `basic` | Top-k cosine similarity search |
| `multi_query` | Expands your query into multiple prompts to increase recall |
| `compression` | Summarizes context chunks before retrieval to help with long docs |

### 🔹 Number of Chunks (k)

How many top chunks to retrieve for each query.  
More chunks = more context, but also slower and more expensive to process.

### 🔹 Chain Type

| Type | Description |
|------|-------------|
| `stuff` | Concatenate all retrieved chunks into a single prompt (fastest) |
| `map_reduce` | LLM runs on each chunk and then summarizes the results |
| `refine` | Iteratively refines the answer with each chunk |
| `map_rerank` | Scores each chunk and returns the best answer |

---

## 🔄 RAG Pipeline Overview

1. **Preprocessing & Cleaning**  
   - Extract raw text from the uploaded PDF or text.  
   - Normalize line breaks, whitespace, remove duplicates.

2. **Text Splitting**  
   - Use your chosen splitter to break the text into overlapping chunks.  
   - **De-duplication**: identical or highly similar chunks (≥ 80% similarity via `difflib.SequenceMatcher`) are dropped to avoid redundancy.

3. **Embedding & Vector Store**  
   - Embed each chunk using your selected model.  
   - Store embeddings in Chroma or FAISS for efficient nearest-neighbor search.

4. **Query Rewriting**  
   - A small LLM chain takes your original question + a short document summary to produce a **refined query**.  
   - Why? Focused queries surface more relevant chunks during retrieval.

5. **Retrieval**  
   - Depending on your retriever choice:  
     - **Basic**: standard top-k cosine similarity.  
     - **Multi-Query**: generate expanded queries for diversity.  
     - **Compression**: summarize and compress chunks before similarity search.

6. **Answer Generation**  
   - The refined query feeds into a `RetrievalQA` chain with your chosen `chain_type`.  
   - Produces the final answer, which you see in the chat interface, along with the top chunks it used.

---

## 📄 Example Session

1. **Setup**  
   - Upload `extension://bfdogplmndidlpjfhoijckpakkdjkkil/pdf/viewer.html?file=https%3A%2F%2Fwww.gptaiflow.tech%2Fassets%2Ffiles%2F2025-01-18-pdf-1-TechAI-Goolge-whitepaper_Prompt%2520Engineering_v4-af36dcc7a49bb7269a58b1c9b89a8ae1.pdf`.  

   - Split: `RecursiveCharacterTextSplitter`  
   - Embedding: `all-MiniLM-L6-v2`  
   - Vector store: `Chroma`  
   - Retriever: `Basic`  
   - k = 15  
   - Chain: `stuff`

2. **Chat**  
   ```text
   User: Which prompt types are there?
   ```
   - LLM refines your question for better search.  
   - Retrieves 5 top chunks.  
   - Runs a map-reduce chain to summarize findings.  
   - Returns a concise answer plus the relevant chunk excerpts.

---

## 🛠️ Extending & Troubleshooting

- **Add new splitters**  
  In `rag.py`, update the `splitter_classes` dict.

- **Swap in a different LLM**  
  Replace `OllamaLLM(model="deepseek-r1:32b")` with your preferred provider.

- **Persisting databases**  
  Chroma persists under `./chroma_db_<model>`. Delete the folder to reset.

- **Errors reading PDFs**  
  Ensure the `PyMuPDF` (`fitz`) version matches your Python. Check stack traces in Streamlit logs.

---

## 📜 License

MIT © Adnan Altukleh
Feel free to adapt this demo for your own RAG experiments!
