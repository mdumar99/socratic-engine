"""
Sources — load text from files, URLs, and plain strings into KG nodes.
"""

import hashlib
import re
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError


def _chunk_text(text: str, chunk_size: int = 200, overlap: int = 20) -> list[str]:
    """Split text into overlapping chunks of roughly chunk_size words."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def _clean_text(text: str) -> str:
    """Remove excessive whitespace and non-printable characters."""
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _stable_id(text: str, offset: int = 0) -> int:
    """Generate a stable numeric ID from text content."""
    h = int(hashlib.md5(text.encode()).hexdigest(), 16)
    return (h % (10**12)) + offset


def nodes_from_text(
    text: str,
    label_prefix: str = "chunk",
    chunk_size: int = 200,
    overlap: int = 20,
    base_id: int = 1000,
) -> list[dict]:
    """Convert raw text into a list of KG node dicts."""
    text = _clean_text(text)
    chunks = _chunk_text(text, chunk_size, overlap)
    nodes = []
    for i, chunk in enumerate(chunks):
        node_id = base_id + i
        label = f"{label_prefix}_{i}"
        nodes.append(
            {
                "id": node_id,
                "label": label,
                "content": chunk[:900],  # stay within MAX_CONTENT_LEN
                "confidence": 0.85,
            }
        )
    return nodes


def nodes_from_file(
    path: str | Path,
    chunk_size: int = 200,
    base_id: int = 1000,
) -> list[dict]:
    """Load a text or markdown file and return KG node dicts."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    text = path.read_text(encoding="utf-8", errors="ignore")
    label_prefix = path.stem.replace(" ", "_")[:30]
    return nodes_from_text(text, label_prefix, chunk_size=chunk_size, base_id=base_id)


def nodes_from_url(
    url: str,
    chunk_size: int = 200,
    base_id: int = 2000,
    timeout: int = 10,
) -> list[dict]:
    """Fetch a URL and return KG node dicts from its text content."""
    try:
        with urlopen(url, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
    except URLError as e:
        raise RuntimeError(f"Failed to fetch {url}: {e}")

    # Strip HTML tags if present
    text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"&\w+;", " ", text)
    return nodes_from_text(
        text, label_prefix="web", chunk_size=chunk_size, base_id=base_id
    )


def nodes_from_string(
    text: str,
    label: str = "custom",
    base_id: int = 3000,
    chunk_size: int = 200,
) -> list[dict]:
    """Convert a plain string directly into KG node dicts."""
    return nodes_from_text(
        text, label_prefix=label, chunk_size=chunk_size, base_id=base_id
    )
