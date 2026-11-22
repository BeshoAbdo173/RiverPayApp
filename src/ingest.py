"""Header-aware markdown chunker, temporal metadata tagger, and ChromaDB indexer."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import settings


H1_RE = re.compile(r"^#\s+(.+?)\s*$")
H2_RE = re.compile(r"^##\s+(.+?)\s*$")
DOC_ID_RE = re.compile(r"\*\*Document ID:\*\*\s*([^\s\n]+)")
EFFECTIVE_RE = re.compile(r"(?:\*\*Effective:\*\*|Effective)\s*(\d{4}-\d{2}-\d{2})")


def _clean_markdown_text(text: str) -> str:
    """Remove HTML aside tags while preserving inner text."""
    text = re.sub(r"</?aside>", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_doc_metadata(filename: str, raw_text: str) -> dict[str, Any]:
    """Extract document-level metadata including temporal and archive flags."""
    doc_id_match = DOC_ID_RE.search(raw_text)
    effective_match = EFFECTIVE_RE.search(raw_text)

    is_archive = "ARCHIVE" in filename.upper() or "ARCHIVE — do not use" in raw_text
    if filename.startswith("08_assistant_policy"):
        doc_type = "policy"
    elif is_archive:
        doc_type = "archive"
    else:
        doc_type = "kb"

    return {
        "doc_id": doc_id_match.group(1) if doc_id_match else filename,
        "effective_date": effective_match.group(1) if effective_match else "1970-01-01",
        "is_archive": is_archive,
        "doc_type": doc_type,
    }


def chunk_markdown_file(file_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Split a markdown file by H1/H2 headings and return chunks + valid section names."""
    raw_text = file_path.read_text(encoding="utf-8")
    filename = file_path.name
    doc_meta = extract_doc_metadata(filename, raw_text)

    lines = raw_text.splitlines()
    chunks: list[dict[str, Any]] = []
    valid_sections: list[str] = []

    current_section: str | None = None
    current_lines: list[str] = []

    def flush_chunk() -> None:
        if current_section is None:
            return
        body = _clean_markdown_text("\n".join(current_lines))
        if not body:
            return
        if current_section not in valid_sections:
            valid_sections.append(current_section)

        status_label = "ARCHIVE (HISTORICAL ONLY)" if doc_meta["is_archive"] else "CURRENT"
        contextual_text = (
            f"[Document: {filename} | Section: {current_section} | "
            f"Effective: {doc_meta['effective_date']} | Status: {status_label}]\n"
            f"{body}"
        )
        chunks.append(
            {
                "id": f"{filename}::{len(chunks):02d}",
                "text": contextual_text,
                "raw_body": body,
                "metadata": {
                    "doc": filename,
                    "section": current_section,
                    "doc_id": doc_meta["doc_id"],
                    "effective_date": doc_meta["effective_date"],
                    "is_archive": doc_meta["is_archive"],
                    "doc_type": doc_meta["doc_type"],
                },
            }
        )

    for line in lines:
        h1_match = H1_RE.match(line)
        h2_match = H2_RE.match(line)
        if h1_match:
            flush_chunk()
            current_section = h1_match.group(1).strip()
            current_lines = []
        elif h2_match:
            flush_chunk()
            current_section = h2_match.group(1).strip()
            current_lines = []
        else:
            if current_section is None:
                current_section = filename
            current_lines.append(line)

    flush_chunk()
    return chunks, valid_sections


def ingest_knowledge_base() -> dict[str, list[str]]:
    """Parse all markdown files in data/, build ChromaDB index, and save valid_sections.json."""
    data_path = Path(settings.data_dir)
    output_path = Path(settings.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    md_files = sorted(data_path.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No markdown files found in {data_path}")

    all_chunks: list[dict[str, Any]] = []
    valid_sections_map: dict[str, list[str]] = {}

    for md_file in md_files:
        file_chunks, file_sections = chunk_markdown_file(md_file)
        all_chunks.extend(file_chunks)
        valid_sections_map[md_file.name] = file_sections

    valid_sections_file = output_path / "valid_sections.json"
    valid_sections_file.write_text(
        json.dumps(valid_sections_map, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    try:
        embedder = SentenceTransformer(settings.embedding_model, local_files_only=True)
    except Exception:
        embedder = SentenceTransformer(settings.embedding_model)
    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts, normalize_embeddings=True).tolist()

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    try:
        client.delete_collection(name=settings.chroma_collection)
    except Exception:
        pass

    collection = client.create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=[c["id"] for c in all_chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[c["metadata"] for c in all_chunks],
    )

    print(
        f"Ingested {len(all_chunks)} chunks across {len(md_files)} files into "
        f"ChromaDB ({settings.chroma_persist_dir}) and saved {valid_sections_file}."
    )
    return valid_sections_map


if __name__ == "__main__":
    ingest_knowledge_base()
