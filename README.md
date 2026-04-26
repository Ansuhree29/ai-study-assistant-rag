# AI Study Assistant (RAG-Based, Fully Local)

## 🚀 Overview
This project is a Retrieval-Augmented Generation (RAG) based AI assistant that answers questions using fully local LLMs.

Built a fully local AI system with zero API dependency using RAG architecture.

## 🛠 Tech Stack
- LangChain
- FAISS
- Ollama (LLaMA 3 / Mistral)
- Sentence Transformers

## ✨ Features
- Fully offline (no OpenAI / API cost)
- Semantic search over documents
- Context-aware answers using RAG
- Fast local inference

## ⚙️ Setup
```bash
pip install -r requirements.txt
ollama pull llama3
python rag_pipeline.py
```

## 📂 Usage
- Run the script:
```bash
python rag_pipeline.py
```
- Enter your query in the terminal

## 📈 Impact
- Eliminated API costs (100% local system)
- Improved privacy (no external data sharing)
- Efficient document retrieval using FAISS
