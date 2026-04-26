"""
RAG Pipeline — Core engine of the AI Study Assistant.
Fixed imports for newer LangChain versions.
"""

import os
import json
import re
import logging
import shutil
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

FAISS_INDEX_DIR = "data/faiss_indexes"
METADATA_DIR    = "data/metadata"


class RAGPipeline:
    def __init__(self):
        self.indexes: Dict[str, Any] = {}
        self.metadata: Dict[str, Dict] = {}
        self.embeddings = None

    def initialize(self):
        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        os.makedirs(METADATA_DIR,    exist_ok=True)
        self._init_embeddings()
        self._load_existing_indexes()

    def _init_embeddings(self):
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or api_key == "your_openai_api_key_here":
            logger.warning("OPENAI_API_KEY not set. Embeddings will be initialised on first use.")
            return
        try:
            from langchain_openai import OpenAIEmbeddings
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=api_key,
                model="text-embedding-3-small",
            )
            logger.info("OpenAI embeddings ready (text-embedding-3-small)")
        except Exception as e:
            logger.error(f"Failed to init OpenAI embeddings: {e}")

    def _get_embeddings(self):
        if self.embeddings:
            return self.embeddings
        self._init_embeddings()
        if not self.embeddings:
            raise RuntimeError(
                "Embeddings not initialised. Check OPENAI_API_KEY in your .env file."
            )
        return self.embeddings

    def _load_existing_indexes(self):
        if not os.path.exists(FAISS_INDEX_DIR):
            return
        for doc_id in os.listdir(FAISS_INDEX_DIR):
            index_path    = os.path.join(FAISS_INDEX_DIR, doc_id)
            metadata_path = os.path.join(METADATA_DIR, f"{doc_id}.json")
            if not os.path.isdir(index_path):
                continue
            try:
                from langchain_community.vectorstores import FAISS
                self.indexes[doc_id] = FAISS.load_local(
                    index_path,
                    self._get_embeddings(),
                    allow_dangerous_deserialization=True,
                )
                if os.path.exists(metadata_path):
                    with open(metadata_path) as f:
                        self.metadata[doc_id] = json.load(f)
                logger.info(f"Loaded index: {doc_id}")
            except Exception as e:
                logger.error(f"Could not load index {doc_id}: {e}")

    def index_document(self, doc_id, chunks, filename):
        from langchain_community.vectorstores import FAISS
        # Fixed: use langchain_core instead of langchain.schema
        try:
            from langchain_core.documents import Document
        except ImportError:
            from langchain.docstore.document import Document

        embeddings = self._get_embeddings()

        documents = [
            Document(
                page_content=chunk["content"],
                metadata={
                    "chunk_id": chunk["chunk_id"],
                    "source":   chunk["source"],
                    "page":     chunk.get("page", 1),
                    "doc_id":   doc_id,
                    "word_count": chunk.get("word_count", 0),
                },
            )
            for chunk in chunks
        ]

        logger.info(f"Generating embeddings for {len(documents)} chunks...")
        vectorstore = FAISS.from_documents(documents, embeddings)

        index_path = os.path.join(FAISS_INDEX_DIR, doc_id)
        vectorstore.save_local(index_path)

        meta = {
            "doc_id":    doc_id,
            "filename":  filename,
            "num_chunks": len(chunks),
        }
        with open(os.path.join(METADATA_DIR, f"{doc_id}.json"), "w") as f:
            json.dump(meta, f, indent=2)

        self.indexes[doc_id]  = vectorstore
        self.metadata[doc_id] = meta

        logger.info(f"Indexed {len(documents)} chunks for '{filename}' (id={doc_id})")
        return len(documents)

    def retrieve(self, query, doc_id=None, top_k=5):
        if not self.indexes:
            return []

        if doc_id and doc_id in self.indexes:
            vectorstore = self.indexes[doc_id]
        else:
            vectorstore = list(self.indexes.values())[-1]

        results = vectorstore.similarity_search_with_score(query, k=top_k)

        chunks = []
        for doc, l2_dist in results:
            similarity = float(1.0 / (1.0 + l2_dist))
            chunks.append({
                "chunk_id":         doc.metadata.get("chunk_id", "unknown"),
                "content":          doc.page_content,
                "source":           doc.metadata.get("source", "unknown"),
                "page":             doc.metadata.get("page", 1),
                "similarity_score": round(similarity, 4),
            })

        chunks.sort(key=lambda x: x["similarity_score"], reverse=True)
        return chunks

    def generate_answer(self, question, chunks, model_type="openai", chat_history=None):
        context_parts = []
        for i, c in enumerate(chunks[:5], 1):
            context_parts.append(
                f"[Chunk {i} | Source: {c['source']}, Page {c['page']} | "
                f"Relevance: {c['similarity_score']:.0%}]\n{c['content']}"
            )
        context = "\n\n" + "\n\n---\n\n".join(context_parts)

        history_str = ""
        if chat_history:
            recent = chat_history[-6:]
            for msg in recent:
                role = "Human" if msg["role"] == "user" else "Assistant"
                history_str += f"{role}: {msg['content']}\n"

        prompt = f"""You are an expert academic study assistant.
Answer the student's question using ONLY the provided document context below.

DOCUMENT CONTEXT:
{context}

{f"CONVERSATION HISTORY:{chr(10)}{history_str}" if history_str else ""}

STUDENT'S QUESTION: {question}

INSTRUCTIONS:
- Answer based solely on the context above
- If context is insufficient, say: "The document doesn't contain enough information about this."
- Cite sources using: (Source: filename, Page X)
- Use bullet points or numbered lists where helpful
- Be clear, detailed, and academically accurate
- Highlight key terms in **bold**

ANSWER:"""

        answer, tokens = self._call_llm(prompt, model_type)

        if chunks:
            scores = [c["similarity_score"] for c in chunks]
            weights = [1.0 / (i + 1) for i in range(len(scores))]
            weighted_avg = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            confidence = round(min(weighted_avg * 1.4, 1.0), 3)
        else:
            confidence = 0.0

        return answer, confidence, tokens

    def _call_llm(self, prompt, model_type):
        if model_type == "openai":
            return self._call_openai(prompt)
        elif model_type == "ollama":
            return self._call_ollama(prompt)
        else:
            raise ValueError(f"Unknown model_type: '{model_type}'")

    def _call_openai(self, prompt):
        from langchain_openai import ChatOpenAI
        # Fixed: use langchain_core instead of langchain.schema
        try:
            from langchain_core.messages import HumanMessage
        except ImportError:
            from langchain.schema import HumanMessage

        llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.2,
            max_tokens=1500,
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            tokens = response.usage_metadata.get("total_tokens", 0)
        return response.content, tokens

    def _call_ollama(self, prompt):
        from langchain_community.llms import Ollama
        llm = Ollama(
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.2,
        )
        return llm.invoke(prompt), 0

    def generate_summary(self, doc_id, model_type="openai"):
        if doc_id not in self.indexes:
            raise ValueError(f"Document '{doc_id}' not found.")

        vectorstore = self.indexes[doc_id]
        sample_queries = [
            "introduction overview main topics",
            "key concepts definitions methods",
            "conclusions results findings summary",
        ]
        seen_content = set()
        all_docs = []

        for q in sample_queries:
            results = vectorstore.similarity_search(q, k=5)
            for doc in results:
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_docs.append(doc)

        context = "\n\n".join(doc.page_content for doc in all_docs[:12])

        prompt = f"""Analyze the following document content and produce a structured study guide.

DOCUMENT CONTENT:
{context}

Respond ONLY with valid JSON (no markdown, no backticks) in this exact format:
{{
    "summary": "A thorough 3-4 paragraph summary covering all major themes and findings.",
    "key_points": [
        "Key point 1",
        "Key point 2",
        "Key point 3",
        "Key point 4",
        "Key point 5",
        "Key point 6",
        "Key point 7",
        "Key point 8"
    ]
}}"""

        raw, _ = self._call_llm(prompt, model_type)

        try:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            result = json.loads(json_match.group()) if json_match else {}
        except (json.JSONDecodeError, AttributeError):
            result = {"summary": raw, "key_points": []}

        return result

    def generate_flashcards(self, doc_id, num_cards=10, model_type="openai"):
        if doc_id not in self.indexes:
            raise ValueError(f"Document '{doc_id}' not found.")

        vectorstore = self.indexes[doc_id]
        docs = vectorstore.similarity_search(
            "key concepts definitions facts important details", k=15
        )
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""You are creating study flashcards for a student.
Generate exactly {num_cards} flashcards from the document content below.

DOCUMENT CONTENT:
{context}

Respond ONLY with a valid JSON array (no markdown, no backticks):
[
  {{"question": "What is ...?", "answer": "..."}},
  {{"question": "Explain ...", "answer": "..."}}
]"""

        raw, _ = self._call_llm(prompt, model_type)

        try:
            json_match = re.search(r"\[.*\]", raw, re.DOTALL)
            cards = json.loads(json_match.group()) if json_match else []
        except (json.JSONDecodeError, AttributeError):
            cards = []

        return cards[:num_cards]

    def get_document_list(self):
        return [
            {
                "doc_id":    doc_id,
                "filename":  meta.get("filename", "Unknown"),
                "num_chunks": meta.get("num_chunks", 0),
            }
            for doc_id, meta in self.metadata.items()
        ]

    def delete_document(self, doc_id):
        if doc_id not in self.indexes:
            return False

        del self.indexes[doc_id]
        del self.metadata[doc_id]

        index_path    = os.path.join(FAISS_INDEX_DIR, doc_id)
        metadata_path = os.path.join(METADATA_DIR, f"{doc_id}.json")

        if os.path.exists(index_path):
            shutil.rmtree(index_path)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)

        logger.info(f"Deleted document {doc_id}")
        return True


rag_pipeline = RAGPipeline()
