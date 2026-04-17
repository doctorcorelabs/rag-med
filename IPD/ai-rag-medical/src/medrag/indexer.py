import hashlib
import html
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import DEFAULT_DB_PATH, DEFAULT_CHROMA_PATH, DEFAULT_WORKSPACE_ROOT, MATERI_GLOB, EMBEDDING_MODEL
from .models import ChunkRecord, ImageRecord, SourcePage

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
IMAGE_RE = re.compile(r"!\[(.*?)\]\((.*?)\)")
WHITESPACE_RE = re.compile(r"\s+")
TABLE_ROW_RE = re.compile(r"^\s*\|.+\|", re.MULTILINE)
TABLE_SEP_RE = re.compile(r"^\s*\|[\s:|-]+\|", re.MULTILINE)
NUMBERED_LIST_RE = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
DOSE_BLOCK_RE = re.compile(r"\d+\s*(?:mg|mcg|µg|unit|ml|g/dl|g/dL|mmol|mEq|IU|gram|tablet|kapsul|ampul|vial)", re.IGNORECASE)

# Section rules for KG edge extraction
SECTION_CATEGORY_MAP = {
    "definisi": "Definisi",
    "pengertian": "Definisi",
    "etiologi": "Etiologi",
    "faktor risiko": "Etiologi",
    "penyebab": "Etiologi",
    "patogenesis": "Patogenesis",
    "patofisiologi": "Patogenesis",
    "mekanisme": "Patogenesis",
    "fisiologi": "Patogenesis",
    "anamnesis": "Anamnesis",
    "anamnesa": "Anamnesis",
    "riwayat": "Anamnesis",
    "manifestasi": "Manifestasi_Klinis",
    "gejala": "Manifestasi_Klinis",
    "klinis": "Manifestasi_Klinis",
    "keluhan": "Manifestasi_Klinis",
    "pemeriksaan fisik": "Pemeriksaan_Fisik",
    "diagnosis": "Diagnosis",
    "pemeriksaan": "Diagnosis",
    "penunjang": "Diagnosis",
    "laboratorium": "Diagnosis",
    "radiologi": "Diagnosis",
    "tata laksana": "Tatalaksana",
    "terapi": "Tatalaksana",
    "pengobatan": "Tatalaksana",
    "obat": "Tatalaksana",
    "farmakologi": "Tatalaksana",
    "dosis": "Tatalaksana",
    "regimen": "Tatalaksana",
    "komplikasi": "Komplikasi",
    "prognosis": "Prognosis",
}


def _checksum(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_text(value: str) -> str:
    decoded = html.unescape(value.replace("&nbsp;", " "))
    return WHITESPACE_RE.sub(" ", decoded).strip()


def discover_source_pages(workspace_root: Path) -> list[SourcePage]:
    pages: list[SourcePage] = []
    for md_path in workspace_root.glob(MATERI_GLOB):
        source_name = md_path.parents[2].name
        page_dir = md_path.parent.name
        page_no = int(page_dir.replace("page-", "")) if page_dir.startswith("page-") else -1
        pages.append(SourcePage(source_name=source_name, page_no=page_no, markdown_path=md_path))
    pages.sort(key=lambda item: (item.source_name, item.page_no))
    return pages


def _split_sections(markdown: str) -> list[tuple[str, str, int, str]]:
    """Split markdown into sections, returning (heading, content, level, parent_heading)."""
    matches = list(HEADING_RE.finditer(markdown))
    if not matches:
        return [("General", markdown, 1, "")]

    sections: list[tuple[str, str, int, str]] = []
    parent_stack: list[tuple[int, str]] = []

    for idx, match in enumerate(matches):
        level = len(match.group(1))
        heading = _normalize_text(match.group(2)) or "General"
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown)
        content = markdown[start:end]

        while parent_stack and parent_stack[-1][0] >= level:
            parent_stack.pop()
        parent_heading = parent_stack[-1][1] if parent_stack else ""
        parent_stack.append((level, heading))

        sections.append((heading, content, level, parent_heading))
    return sections


@dataclass
class _SemanticBlock:
    text: str
    block_type: str  # "prose" | "table" | "list" | "dose_block"


def _split_into_semantic_blocks(text: str) -> list[_SemanticBlock]:
    """Split text into semantic blocks: tables, numbered lists, dose blocks, and prose."""
    lines = text.split("\n")
    blocks: list[_SemanticBlock] = []
    current_lines: list[str] = []
    current_type = "prose"

    def _flush():
        nonlocal current_lines, current_type
        body = "\n".join(current_lines).strip()
        if body:
            blocks.append(_SemanticBlock(text=body, block_type=current_type))
        current_lines = []
        current_type = "prose"

    i = 0
    while i < len(lines):
        line = lines[i]
        is_table = bool(TABLE_ROW_RE.match(line))
        is_numbered = bool(NUMBERED_LIST_RE.match(line))

        if is_table:
            if current_type != "table":
                _flush()
                current_type = "table"
            current_lines.append(line)
        elif is_numbered:
            if current_type != "list":
                _flush()
                current_type = "list"
            current_lines.append(line)
        else:
            if current_type in ("table", "list"):
                if not line.strip():
                    current_lines.append(line)
                    _flush()
                else:
                    _flush()
                    current_type = "prose"
                    current_lines.append(line)
            else:
                current_lines.append(line)
        i += 1

    _flush()

    final_blocks: list[_SemanticBlock] = []
    for block in blocks:
        if block.block_type == "prose" and DOSE_BLOCK_RE.search(block.text):
            block = _SemanticBlock(text=block.text, block_type="dose_block")
        final_blocks.append(block)

    return final_blocks


def _chunk_text(
    value: str,
    max_chars: int = 1800,
    table_max: int = 2500,
    heading: str = "",
    source_name: str = "",
    page_no: int = 0,
) -> list[tuple[str, str]]:
    """Returns list of (chunk_content, content_type) tuples."""
    text = value.strip()
    if not text:
        return []

    metadata_prefix = f"[Konteks: {source_name} - Hal {page_no} | Topik Utama: {heading}]\n"

    semantic_blocks = _split_into_semantic_blocks(text)
    chunks: list[tuple[str, str]] = []

    current_chunk = ""
    current_type = "prose"
    last_paragraph = ""

    def _push_chunk(content: str, ctype: str):
        normalized = _normalize_text(content) if ctype == "prose" else content.strip()
        if normalized:
            chunks.append((metadata_prefix + normalized, ctype))

    for block in semantic_blocks:
        budget = table_max if block.block_type == "table" else max_chars
        max_payload = budget - len(metadata_prefix)

        if block.block_type in ("table", "list", "dose_block"):
            if current_chunk.strip():
                _push_chunk(current_chunk, current_type)
                current_chunk = ""
                last_paragraph = ""

            if len(block.text) <= max_payload:
                _push_chunk(block.text, block.block_type)
            else:
                lines = block.text.split("\n")
                sub = ""
                for line in lines:
                    if len(sub) + len(line) + 1 > max_payload and sub:
                        _push_chunk(sub, block.block_type)
                        sub = ""
                    sub += line + "\n"
                if sub.strip():
                    _push_chunk(sub, block.block_type)
            continue

        max_payload = max_chars - len(metadata_prefix)
        paragraphs = re.split(r"\n\s*\n", block.text)
        current_type = "prose"

        for p in paragraphs:
            if len(current_chunk) + len(p) > max_payload and current_chunk:
                _push_chunk(current_chunk, current_type)
                if len(last_paragraph) < (max_payload * 0.25):
                    current_chunk = last_paragraph + "\n\n"
                else:
                    current_chunk = ""

            if len(p) >= max_payload:
                if current_chunk:
                    _push_chunk(current_chunk, current_type)
                    current_chunk = ""
                lines = p.split("\n")
                sub = ""
                for line in lines:
                    if len(sub) + len(line) < max_payload:
                        sub += line + "\n"
                    else:
                        if sub:
                            _push_chunk(sub, current_type)
                        sub = line + "\n"
                current_chunk = sub + "\n"
                last_paragraph = ""
            else:
                current_chunk += p + "\n\n"
                last_paragraph = p

    if current_chunk.strip():
        _push_chunk(current_chunk, current_type)

    return chunks


def _derive_tags(heading: str, text: str) -> str:
    seeds = f"{heading} {text[:300]}".lower()
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9/-]{2,}", seeds)
    unique: list[str] = []
    for token in tokens:
        if token not in unique:
            unique.append(token)
        if len(unique) == 12:
            break
    return " ".join(unique)


def _detect_section_category(heading: str) -> str | None:
    """Heuristic detection of which clinical section a heading belongs to."""
    h_lower = heading.lower()
    for keyword, category in SECTION_CATEGORY_MAP.items():
        if keyword in h_lower:
            return category
    return None


def parse_page(source_page: SourcePage) -> tuple[list[ChunkRecord], list[ImageRecord]]:
    markdown = source_page.markdown_path.read_text(encoding="utf-8", errors="ignore")
    sections = _split_sections(markdown)

    chunk_records: list[ChunkRecord] = []
    image_records: list[ImageRecord] = []
    found_image_refs: set[str] = set()

    for heading, raw_content, level, parent_heading in sections:
        for alt_text, image_ref in IMAGE_RE.findall(raw_content):
            image_abs_path = (source_page.markdown_path.parent / image_ref).resolve()
            nearby = _normalize_text(IMAGE_RE.sub(" ", raw_content))[:300]
            checksum_seed = f"{source_page.markdown_path}|{heading}|{image_ref}|{nearby}"
            found_image_refs.add(image_ref)
            image_records.append(
                ImageRecord(
                    source_name=source_page.source_name,
                    page_no=source_page.page_no,
                    alt_text=_normalize_text(alt_text),
                    image_ref=image_ref,
                    image_abs_path=str(image_abs_path),
                    heading=heading,
                    nearby_text=nearby,
                    markdown_path=str(source_page.markdown_path),
                    checksum=_checksum(checksum_seed),
                )
            )

        cleaned = IMAGE_RE.sub(" ", raw_content)
        section_category = _detect_section_category(heading)
        section_chunks = _chunk_text(
            cleaned,
            heading=heading,
            source_name=source_page.source_name,
            page_no=source_page.page_no,
        )
        total_in_section = len(section_chunks)

        for chunk_idx, (chunk_content, content_type) in enumerate(section_chunks):
            checksum_seed = f"{source_page.markdown_path}|{heading}|{chunk_content}"
            chunk_records.append(
                ChunkRecord(
                    source_name=source_page.source_name,
                    page_no=source_page.page_no,
                    heading=heading,
                    content=chunk_content,
                    disease_tags=_derive_tags(heading, chunk_content),
                    markdown_path=str(source_page.markdown_path),
                    checksum=_checksum(checksum_seed),
                    section_category=section_category or "Ringkasan_Klinis",
                    parent_heading=parent_heading,
                    chunk_index=chunk_idx,
                    total_chunks=total_in_section,
                    heading_level=level,
                    content_type=content_type,
                )
            )

    if not found_image_refs:
        for image_path in sorted(source_page.markdown_path.parent.glob("img-*.*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
                continue
            checksum_seed = f"{source_page.markdown_path}|fallback|{image_path.name}"
            image_records.append(
                ImageRecord(
                    source_name=source_page.source_name,
                    page_no=source_page.page_no,
                    alt_text=image_path.name,
                    image_ref=image_path.name,
                    image_abs_path=str(image_path.resolve()),
                    heading="Image from page folder",
                    nearby_text="",
                    markdown_path=str(source_page.markdown_path),
                    checksum=_checksum(checksum_seed),
                )
            )

    return chunk_records, image_records


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode = WAL;

        DROP TABLE IF EXISTS chunks_fts;
        DROP TABLE IF EXISTS chunks;
        DROP TABLE IF EXISTS images;
        DROP TABLE IF EXISTS graph_edges;

        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_name TEXT NOT NULL,
            page_no INTEGER NOT NULL,
            heading TEXT NOT NULL,
            content TEXT NOT NULL,
            disease_tags TEXT NOT NULL,
            markdown_path TEXT NOT NULL,
            checksum TEXT NOT NULL,
            section_category TEXT NOT NULL DEFAULT 'Ringkasan_Klinis',
            parent_heading TEXT NOT NULL DEFAULT '',
            chunk_index INTEGER NOT NULL DEFAULT 0,
            total_chunks INTEGER NOT NULL DEFAULT 1,
            heading_level INTEGER NOT NULL DEFAULT 2,
            content_type TEXT NOT NULL DEFAULT 'prose'
        );

        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            content,
            disease_tags,
            source_name,
            heading,
            content='chunks',
            content_rowid='id'
        );

        CREATE TABLE images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_name TEXT NOT NULL,
            page_no INTEGER NOT NULL,
            alt_text TEXT,
            image_ref TEXT NOT NULL,
            image_abs_path TEXT NOT NULL,
            heading TEXT NOT NULL,
            nearby_text TEXT NOT NULL,
            markdown_path TEXT NOT NULL,
            checksum TEXT NOT NULL
        );

        CREATE TABLE graph_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_disease TEXT NOT NULL,
            relation TEXT NOT NULL,
            target_node TEXT NOT NULL,
            target_type TEXT NOT NULL,
            source_name TEXT NOT NULL,
            page_no INTEGER NOT NULL
        );
        """
    )


def _insert_records(conn: sqlite3.Connection, chunks: list[ChunkRecord], images: list[ImageRecord]) -> None:
    conn.executemany(
        """
        INSERT INTO chunks (source_name, page_no, heading, content, disease_tags, markdown_path, checksum,
                            section_category, parent_heading, chunk_index, total_chunks, heading_level, content_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                record.source_name,
                record.page_no,
                record.heading,
                record.content,
                record.disease_tags,
                record.markdown_path,
                record.checksum,
                record.section_category,
                record.parent_heading,
                record.chunk_index,
                record.total_chunks,
                record.heading_level,
                record.content_type,
            )
            for record in chunks
        ],
    )

    conn.execute(
        """
        INSERT INTO chunks_fts(rowid, content, disease_tags, source_name, heading)
        SELECT id, content, disease_tags, source_name, heading FROM chunks
        """
    )

    conn.executemany(
        """
        INSERT INTO images (source_name, page_no, alt_text, image_ref, image_abs_path, heading, nearby_text, markdown_path, checksum)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                record.source_name,
                record.page_no,
                record.alt_text,
                record.image_ref,
                record.image_abs_path,
                record.heading,
                record.nearby_text,
                record.markdown_path,
                record.checksum,
            )
            for record in images
        ],
    )


def _insert_graph_edges(conn: sqlite3.Connection, chunks: list[ChunkRecord]) -> None:
    """Extract heuristic knowledge graph edges from chunk headings."""
    edges = []
    for chunk in chunks:
        disease_node = chunk.source_name
        category = chunk.section_category
        # Extract key terms from heading as target nodes
        heading_words = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9/ ]{2,}", chunk.heading)
        for term in heading_words[:3]:
            term = term.strip()
            if term and term.lower() != disease_node.lower():
                edges.append((disease_node, category, term, "concept", chunk.source_name, chunk.page_no))

    if edges:
        conn.executemany(
            "INSERT INTO graph_edges (source_disease, relation, target_node, target_type, source_name, page_no) VALUES (?, ?, ?, ?, ?, ?)",
            edges,
        )


def _build_vector_index(chunks: list[ChunkRecord], chroma_path: Path) -> None:
    """Build ChromaDB vector index from chunks using sentence-transformers."""
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[WARN] chromadb or sentence-transformers not installed. Skipping vector index.")
        return

    print("[INFO] Building vector index (this may take a while on first run)...")
    chroma_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(chroma_path))

    # Delete existing collection to rebuild fresh
    try:
        client.delete_collection("medrag_chunks")
    except Exception:
        pass

    collection = client.create_collection(
        name="medrag_chunks",
        metadata={"hnsw:space": "cosine"},
    )

    model = SentenceTransformer(EMBEDDING_MODEL)

    batch_size = 64
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [f"passage: {c.heading}: {c.content}" for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        ids = [f"chunk_{i + j}" for j in range(len(batch))]
        metadatas = [
            {
                "source_name": c.source_name,
                "page_no": c.page_no,
                "heading": c.heading,
                "section_category": c.section_category,
                "disease_tags": c.disease_tags,
                "content": c.content,
                "parent_heading": c.parent_heading,
                "content_type": c.content_type,
                "chunk_index": c.chunk_index,
            }
            for c in batch
        ]
        collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    print(f"[INFO] Vector index built: {len(chunks)} chunks embedded.")


def build_index(
    workspace_root: Path | None = None,
    db_path: Path | None = None,
    chroma_path: Path | None = None,
    skip_vector: bool = False,
) -> dict[str, int]:
    workspace = workspace_root or DEFAULT_WORKSPACE_ROOT
    database = db_path or DEFAULT_DB_PATH
    chroma = chroma_path or DEFAULT_CHROMA_PATH
    database.parent.mkdir(parents=True, exist_ok=True)

    source_pages = discover_source_pages(workspace)

    all_chunks: list[ChunkRecord] = []
    all_images: list[ImageRecord] = []
    for page in source_pages:
        chunks, images = parse_page(page)
        all_chunks.extend(chunks)
        all_images.extend(images)

    conn = sqlite3.connect(database)
    try:
        _init_schema(conn)
        _insert_records(conn, all_chunks, all_images)
        _insert_graph_edges(conn, all_chunks)
        conn.commit()
    finally:
        conn.close()

    if not skip_vector:
        _build_vector_index(all_chunks, chroma)

    return {
        "source_pages": len(source_pages),
        "chunks": len(all_chunks),
        "images": len(all_images),
    }
