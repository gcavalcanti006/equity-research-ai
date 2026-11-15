"""Read text from a PDF page by page and split it into chunks that keep page numbers."""
from dataclasses import dataclass

import pdfplumber


@dataclass
class Chunk:
    index: int
    start_page: int
    end_page: int
    text: str          # pages joined with '=== PAGE n ===' markers


def read_pages(pdf_path):
    """Return a list with the text of each page (list position 0 = page 1)."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return pages


def is_scanned(pages, min_chars):
    """A PDF with almost no extractable text is probably scanned images."""
    return sum(len(p.strip()) for p in pages) < min_chars


def split_into_chunks(pages, chunk_pages, max_chars):
    """Group consecutive pages into chunks of at most `chunk_pages` pages / `max_chars` characters.

    Every page is labelled '=== PAGE n ===' so the model can cite page numbers and we can
    check its quotes against the right page later. Pages with no text are skipped."""
    chunks, current, current_chars, start = [], [], 0, None

    def close():
        nonlocal current, current_chars, start
        if current:
            chunks.append(Chunk(len(chunks), start, current_end, "\n\n".join(current)))
        current, current_chars, start = [], 0, None

    current_end = 0
    for number, text in enumerate(pages, start=1):
        if not text.strip():
            continue
        block = f"=== PAGE {number} ===\n{text.strip()}"
        too_many = len(current) >= chunk_pages
        too_long = current and current_chars + len(block) > max_chars
        if too_many or too_long:
            close()
        if start is None:
            start = number
        current.append(block)
        current_chars += len(block)
        current_end = number
    close()
    return chunks
