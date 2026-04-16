from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SourcePage:
    source_name: str
    page_no: int
    markdown_path: Path


@dataclass(slots=True)
class ChunkRecord:
    source_name: str
    page_no: int
    heading: str
    content: str
    disease_tags: str
    markdown_path: str
    checksum: str
    section_category: str = "Ringkasan_Klinis"


@dataclass(slots=True)
class ImageRecord:
    source_name: str
    page_no: int
    alt_text: str
    image_ref: str
    image_abs_path: str
    heading: str
    nearby_text: str
    markdown_path: str
    checksum: str


@dataclass
class ChatHistoryItem:
    role: str  # "user" or "assistant"
    content: str


@dataclass
class KnowledgeGraphNode:
    id: str
    label: str
    type: str  # "disease" | "concept" | "section"
    group: int = 1


@dataclass
class KnowledgeGraphEdge:
    source: str
    target: str
    relation: str


@dataclass
class KnowledgeGraphResponse:
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    disease: str
