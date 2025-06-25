# 📚 Automated Metadata Generator using RAG, FAISS & Streamlit

This project is a **Retrieval-Augmented Generation (RAG)** application that:
- Answers natural language questions based on the contents of uploaded documents.
- Automatically generates structured metadata in JSON format for each document.

It uses:
- **FAISS** for semantic search over document embeddings
- **Mistral-7B-Instruct** (via Hugging Face API or local `.gguf` model)
- **LangChain** for chaining prompts and models
- **Streamlit** as an interactive web app interface

---

## ✨ Features

- 📄 Upload multiple documents (`.pdf`, `.docx`, `.txt`)
- ❓ Ask questions to get document-aware answers using a RAG pipeline
- 🧠 Automatically extract rich metadata such as title, summary, keywords, and more
- 🔍 Semantic search using vector embeddings with FAISS
- 🖥️ Optional support for running large language models locally via GGUF & llama.cpp

---



## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/madboi11/Automated-Meta-Data-Generation
cd automated-metadata-generator
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your HuggingFaceHub access token here
```bash
set HUGGINGFACEHUB_API_TOKEN=your_token_here
```

### 4. Run the web app
```bash
streamlit run app.py
```




