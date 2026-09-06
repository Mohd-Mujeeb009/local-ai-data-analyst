"""
Structure-aware chunking for PDF text.

Two decisions carry most of the retrieval quality here.

**Split on structure, not on a fixed character count.** A window that cuts
mid-sentence produces chunks that read as fragments and embed as noise.
Headings are the document's own statement about where topics begin.

**Prepend the heading path to every chunk before embedding.** A chunk that says
"revenue fell 12% year over year" is nearly useless on its own - fell for what,
in which segment? Embedded as
"Financial Review > Segment Performance > EMEA: revenue fell 12%..." it carries
the context that makes it retrievable, and a query about EMEA performance can
actually reach it. The heading path is stored separately from the body so the UI
can cite the section without the prefix showing up in the quoted snippet.
"""

import re

# A heading is a short line that is not prose. Real PDFs have no markup, so this
# leans on the shapes that survive text extraction: numbered sections, ALL CAPS
# banners, and short title-case lines with no terminal punctuation.
NUMBERED = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+(\S.{0,80})$")
ALL_CAPS = re.compile(r"^\s*([A-Z][A-Z0-9 &/,'()-]{3,70})\s*$")
MARKDOWN = re.compile(r"^\s*(#{1,6})\s+(\S.*)$")

MAX_HEADING_WORDS = 12
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _heading_level(line, current_depth):
    """
    Classify a line as a heading and return its (level, text), or None.

    Args:
        line: The candidate line.
        current_depth: Nesting level of the enclosing heading, or 0 at the top.

    Returns:
        tuple[int, str] or None: Nesting level and heading text.

    Numbering states its own level, so "3.2.1 Revenue" is unambiguous. Unnumbered
    headings do not: an ALL CAPS "EMEA" under "1.1 Segment Performance" is a
    sub-heading, not a new top-level section. Treating it as level 1 would pop
    the stack and cost the chunk the very context the heading path exists to
    supply, so unnumbered headings nest one level below their enclosing section.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 90:
        return None

    markdown = MARKDOWN.match(stripped)
    if markdown:
        return len(markdown.group(1)), markdown.group(2).strip()

    numbered = NUMBERED.match(stripped)
    if numbered:
        # "3.2.1 Revenue" nests one level deeper than "3.2 Segments".
        return numbered.group(1).count(".") + 1, numbered.group(2).strip()

    if stripped.endswith((".", ",", ";", ":")):
        return None

    nested = current_depth + 1

    if len(stripped.split()) <= MAX_HEADING_WORDS and ALL_CAPS.match(stripped):
        return nested, stripped

    # Short Title Case line with no terminal punctuation.
    words = stripped.split()
    if not words or len(words) > MAX_HEADING_WORDS:
        return None

    # A lone capitalised word on its own line - "Americas", "Overview" - is a
    # heading in extracted PDF text far more often than it is a stray fragment.
    # Missing one is expensive: the section below it merges into its predecessor
    # and retrieval then returns the neighbouring section's figures under this
    # section's name, which is the precise error this module exists to avoid.
    if len(words) == 1:
        word = words[0]
        return (nested, stripped) if word[:1].isupper() and word.isalpha() and len(word) >= 3 else None

    capitalised = sum(1 for w in words if w[:1].isupper())
    if capitalised >= max(2, int(len(words) * 0.6)):
        return nested, stripped

    return None


def _is_numbered(line):
    """Report whether a heading line states its own level via numbering."""
    stripped = line.strip()
    return bool(MARKDOWN.match(stripped) or NUMBERED.match(stripped))


def _split_long_block(text, max_chars, overlap):
    """
    Split an oversized section on sentence boundaries.

    Args:
        text: The section body.
        max_chars: Target maximum characters per piece.
        overlap: Characters of trailing context to repeat into the next piece,
            so a fact spanning a boundary is not lost to both sides.

    Returns:
        list[str]: The pieces.
    """
    sentences = [s for s in SENTENCE_END.split(text) if s.strip()]
    pieces, current = [], ""

    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_chars:
            pieces.append(current.strip())
            tail = current[-overlap:] if overlap else ""
            current = (tail + " " + sentence) if tail else sentence
        else:
            current = (current + " " + sentence) if current else sentence

    if current.strip():
        pieces.append(current.strip())
    return pieces or [text.strip()]


def chunk_document(text, doc_id, max_chars=1200, overlap=150, min_chars=80):
    """
    Split extracted PDF text into retrievable chunks.

    Args:
        text: The full extracted text.
        doc_id: Identifier for the source document.
        max_chars: Target maximum body length per chunk.
        overlap: Characters repeated across a forced split.
        min_chars: Chunks shorter than this are merged into their predecessor,
            which keeps stray headers and page numbers out of the index.

    Returns:
        list[dict]: Each with "id", "doc_id", "heading_path", "body", "text"
            (the heading-prefixed form that gets embedded) and "order".
    """
    lines = text.splitlines()
    stack = []          # active heading path, as (level, title, numbered)
    sections = []       # (heading_path, body)
    body = []

    def flush():
        if any(line.strip() for line in body):
            sections.append(
                (" > ".join(t for _, t, _ in stack), "\n".join(body).strip())
            )

    for line in lines:
        heading = _heading_level(line, stack[-1][0] if stack else 0)
        if heading:
            flush()
            body = []
            level, title = heading
            numbered = _is_numbered(line)

            # Consecutive unnumbered headings are siblings, not ancestors.
            # Without this, "EMEA" then "Americas" would read as Americas
            # nested inside EMEA - a heading path that misdescribes the
            # document and misleads whoever reads the citation.
            if not numbered and stack and not stack[-1][2]:
                level = stack[-1][0]

            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title, numbered))
        else:
            body.append(line)
    flush()

    chunks = []
    for heading_path, section_body in sections:
        collapsed = re.sub(r"\n{2,}", "\n\n", section_body).strip()
        if not collapsed:
            continue

        for piece in _split_long_block(collapsed, max_chars, overlap):
            if not piece.strip():
                continue
            # Merge a runt into its predecessor rather than indexing a fragment.
            if chunks and len(piece) < min_chars and chunks[-1]["heading_path"] == heading_path:
                chunks[-1]["body"] += "\n" + piece
                chunks[-1]["text"] = _embed_form(heading_path, chunks[-1]["body"])
                continue

            chunks.append({
                "id": f"{doc_id}::{len(chunks):04d}",
                "doc_id": doc_id,
                "heading_path": heading_path,
                "body": piece,
                "text": _embed_form(heading_path, piece),
                "order": len(chunks),
            })

    if not chunks:
        # A document with no detectable structure still has to be searchable.
        for piece in _split_long_block(text.strip(), max_chars, overlap):
            chunks.append({
                "id": f"{doc_id}::{len(chunks):04d}",
                "doc_id": doc_id,
                "heading_path": "",
                "body": piece,
                "text": piece,
                "order": len(chunks),
            })

    return chunks


def _embed_form(heading_path, body):
    """Return the text that gets embedded: heading path, then body."""
    return f"{heading_path}: {body}" if heading_path else body
