"""Hash helpers used for file deduplication and chunk identity."""

import hashlib


def md5_text(text: str, encoding: str = "utf-8") -> str:
    return hashlib.md5(text.encode(encoding=encoding)).hexdigest()


def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def chunk_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]
