"""Small file helpers for uploaded enterprise documents."""

from pathlib import Path


def normalize_file_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower().lstrip(".")
    return suffix or "txt"


def safe_filename(filename: str) -> str:
    keep = []
    for char in filename:
        if char.isalnum() or char in ("-", "_", ".", " ", "(", ")"):
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip() or "uploaded_file"


def read_text_with_fallback(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")
