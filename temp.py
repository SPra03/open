# ── requirements ───────────────────────────────────────────────
# pip install langchain tiktoken regex
import re
from typing import List
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ── 0. Helper: token-length function using tiktoken ────────────
enc = tiktoken.encoding_for_model("text-embedding-3-large")

def token_len(text: str) -> int:
    """Return number of tokens in `text` for OpenAI embeddings."""
    return len(enc.encode(text))

# ── 1. Identify atomic VISUAL/TABLE blocks before chunking ─────
VISUAL_RE = re.compile(
    r"\[VISUAL](?:.|\n)*?\[CLINICAL_SIGNIFICANCE](?:.|\n)*?(?=\[PAGE|\[VISUAL]|\[TABLE]|$)",
    flags=re.MULTILINE,
)
TABLE_RE = re.compile(
    r"\[TABLE](?:.|\n)*?(?=\[PAGE|\[VISUAL]|\[TABLE]|$)",
    flags=re.MULTILINE,
)

def carve_atomic_blocks(raw: str) -> List[str]:
    """Return list where VISUAL/TABLE blocks are isolated; everything else remains."""
    spans = []
    last = 0
    for m in sorted(
        list(VISUAL_RE.finditer(raw)) + list(TABLE_RE.finditer(raw)),
        key=lambda x: x.start(),
    ):
        # narrative before the atomic block
        if m.start() > last:
            spans.append(raw[last : m.start()])
        # the atomic block itself
        spans.append(m.group(0))
        last = m.end()
    # tail narrative
    if last < len(raw):
        spans.append(raw[last:])
    return [s.strip() for s in spans if s.strip()]

# ── 2. Tag‑aware recursive splitter for narrative text ─────────
SEPARATORS = [
    "[PAGE", "[SECTION:", "[SUBSECTION:", "[VISUAL]", "[TABLE]",
    "\n\n", "\n", ".", "?", "!", " "
]

narrative_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=40,           # ≈10 % of 400
    separators=SEPARATORS,
    length_function=token_len,
)

# ── 3. Table‑row splitter for giant tables (>1 k tokens) ───────
def split_table_rows(tbl: str, max_tokens: int = 1000) -> List[str]:
    """Split a big [TABLE] block by rows, copying the header JSON to each slice."""
    if token_len(tbl) <= max_tokens:
        return [tbl]

    # crude row split at `"},{` boundaries in the JSON block
    header, *rows = tbl.split('},{')
    header += '},'  # re‑append the brace removed above
    chunks, current = [], header
    for row in rows:
        row_chunk = row + ('},{' if row is not rows[-1] else '')
        if token_len(current + row_chunk) > max_tokens:
            chunks.append(current.rstrip(',{'))
            current = header + row_chunk
        else:
            current += row_chunk
    chunks.append(current.rstrip(',{'))
    return chunks

# ── 4. Post‑filter to merge tiny narrative chunks (<120 tokens) ─
def merge_micro_chunks(chunks: List[str]) -> List[str]:
    output = []
    for c in chunks:
        if token_len(c) < 120 and output:
            output[-1] = output[-1].rstrip() + "\n" + c.lstrip()
        else:
            output.append(c)
    return output

# ── 5. Main entry point ────────────────────────────────────────
def chunk_raw_document(raw_text: str) -> List[str]:
    final_chunks = []
    for block in carve_atomic_blocks(raw_text):
        if block.startswith("[VISUAL]"):
            final_chunks.append(block)            # atomic, no overlap
        elif block.startswith("[TABLE]"):
            final_chunks.extend(split_table_rows(block))
        else:                                     # narrative
            final_chunks.extend(narrative_splitter.split_text(block))
    # merge very small trailing pieces
    final_chunks = merge_micro_chunks(final_chunks)
    return final_chunks

# ── example usage ──────────────────────────────────────────────
if __name__ == "__main__":
    with open("zeposia_raw.txt", "r", encoding="utf‑8") as f:
        raw = f.read()
    chunks = chunk_raw_document(raw)
    print(f"Produced {len(chunks)} chunks; first 3 preview:\n")
    for c in chunks[:3]:
        print("-"*80)
        print(c[:500], "\n…\n")
