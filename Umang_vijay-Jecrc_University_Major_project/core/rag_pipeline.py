"""
RAG Pipeline — FAISS-backed Retrieval-Augmented Generation for self-correction.
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document


class RAGPipeline:
    """RAG self-correction engine using FAISS vector store over Pandas/Python docs."""

    def __init__(self, api_key: str, docs_dir: Optional[str] = None,
                 index_dir: Optional[str] = None):
        self.api_key = api_key
        self.project_root = Path(__file__).parent.parent
        self.docs_dir = Path(docs_dir) if docs_dir else self.project_root / "docs_corpus"
        self.index_dir = Path(index_dir) if index_dir else self.project_root / "faiss_index"

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key,
        )
        self.vector_store: Optional[FAISS] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def build_or_load_index(self) -> bool:
        """Build FAISS index from docs or load from disk. Returns True if ready."""
        if self.index_dir.exists() and (self.index_dir / "index.faiss").exists():
            try:
                self.vector_store = FAISS.load_local(
                    str(self.index_dir), self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                return True
            except Exception:
                pass
        return self.build_index()

    def build_index(self) -> bool:
        """Build FAISS index from all .txt files in docs_dir."""
        if not self.docs_dir.exists():
            return False

        documents = []
        for txt_file in sorted(self.docs_dir.glob("*.txt")):
            try:
                content = txt_file.read_text(encoding="utf-8")
                if content.strip():
                    documents.append(Document(
                        page_content=content,
                        metadata={"source": txt_file.name,
                                  "category": txt_file.stem.replace("_", " ").title()},
                    ))
            except Exception:
                continue

        if not documents:
            return False

        chunks = self.text_splitter.split_documents(documents)
        try:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(str(self.index_dir))
            return True
        except Exception:
            return False

    def query_for_fix(self, error_message: str, original_code: str, top_k: int = 5) -> str:
        """Query docs to find relevant context for fixing a code error."""
        if self.vector_store is None:
            return "RAG index not available."

        query = self._build_error_query(error_message, original_code)
        try:
            results = self.vector_store.similarity_search_with_score(query, k=top_k)
        except Exception as e:
            return f"RAG query failed: {e}"

        if not results:
            return "No relevant documentation found."

        parts = []
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source", "unknown")
            parts.append(
                f"--- Doc Snippet {i} (source: {source}, score: {score:.3f}) ---\n"
                f"{doc.page_content.strip()}\n"
            )
        return "\n\n".join(parts)

    def query(self, question: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """General-purpose query. Returns list of (content, score)."""
        if self.vector_store is None:
            return []
        try:
            results = self.vector_store.similarity_search_with_score(question, k=top_k)
            return [(doc.page_content, score) for doc, score in results]
        except Exception:
            return []

    def _build_error_query(self, error_message: str, code: str) -> str:
        """Build optimized search query from error and code context."""
        error_lines = error_message.strip().split("\n")
        core_error = "\n".join(error_lines[-3:]) if len(error_lines) > 3 else error_message

        error_type = ""
        for line in reversed(error_lines):
            if "Error" in line and ":" in line:
                error_type = line.split(":")[0].strip().split(".")[-1]
                break

        parts = []
        if error_type:
            parts.append(f"Python {error_type}")
        parts.append(core_error[:300])

        pandas_ops = re.findall(r'\.(\w+)\(', code)
        if pandas_ops:
            parts.append(f"pandas: {', '.join(list(set(pandas_ops))[:5])}")

        return " ".join(parts)

    def get_index_stats(self) -> dict:
        """Get statistics about the current FAISS index."""
        if self.vector_store is None:
            return {"status": "not_loaded", "num_vectors": 0}
        try:
            return {"status": "loaded", "num_vectors": self.vector_store.index.ntotal,
                    "index_dir": str(self.index_dir)}
        except Exception:
            return {"status": "error", "num_vectors": 0}
