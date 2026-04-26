# ai-study-assistant-rag
RAG-based AI Study Assistant using LangChain, FAISS, and Ollama with fully local embeddings (no OpenAI dependency)
# AI Study Assistant (RAG-Based, Fully Local)

## 🚀 Overview
This project is a Retrieval-Augmented Generation (RAG) based AI assistant that answers questions from PDFs using fully local LLMs.

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

pip install -r requirements.txt  
ollama pull llama3  
python rag_pipeline.py  

## 📂 Usage
- Place your PDF inside a `data/` folder
- Run the script
- Ask questions in terminal

## 📈 Impact
- Eliminated API costs (100% local system)
- Improved privacy (no external data sharing)
- Efficient document retrieval using FAISS
