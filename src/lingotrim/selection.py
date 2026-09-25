from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import regex
from tokenizers import Tokenizer

from .bytelevel import token_bytes, token_text
from .tokenizer_json import TokenizerJson

KNOWN_SCRIPTS = [
    "Latin", "Cyrillic", "Greek", "Han", "Hiragana", "Katakana", "Hangul", "Arabic", "Hebrew",
    "Thai", "Devanagari", "Bengali", "Tamil", "Telugu", "Georgian", "Armenian", "Ethiopic",
    "Khmer", "Lao", "Myanmar", "Gujarati", "Gurmukhi", "Kannada", "Malayalam", "Sinhala",
]


@dataclass
class SelectionConfig:
    min_freq: int = 1
    keep_ascii: bool = True
    scripts: list[str] = field(default_factory=list)
    extra_ids: set[int] = field(default_factory=set)


@dataclass
class Selection:
    keep: list[int]
    reasons: Counter


def iter_corpus_texts(paths: Iterable[str | Path], chunk_chars: int = 4000) -> Iterator[str]:
    """Yield text chunks from files or directories (recursively, *.txt / *.jsonl / code)."""
    for p in map(Path, paths):
        files = sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p]
        for f in files:
            buf: list[str] = []
            size = 0
            with f.open(encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    buf.append(line)
                    size += len(line)
                    if size >= chunk_chars:
                        yield "".join(buf)
                        buf, size = [], 0
            if buf:
                yield "".join(buf)


def count_corpus_tokens(tokenizer: Tokenizer, texts: Iterable[str], batch: int = 256) -> Counter:
    counts: Counter = Counter()
    pending: list[str] = []
    for text in texts:
        pending.append(text)
        if len(pending) >= batch:
            for enc in tokenizer.encode_batch(pending, add_special_tokens=False):
                counts.update(enc.ids)
            pending.clear()
    if pending:
        for enc in tokenizer.encode_batch(pending, add_special_tokens=False):
            counts.update(enc.ids)
    return counts


def _script_pattern(scripts: list[str]) -> regex.Pattern:
    classes = "".join(rf"\p{{Script={s}}}" for s in [*scripts, "Common", "Inherited"])
    return regex.compile(rf"^[{classes}]+$")


def select_tokens(
    tok: TokenizerJson, corpus_counts: Counter | None, cfg: SelectionConfig
) -> Selection:
    reasons: Counter = Counter()
    keep: set[int] = set()

    def add(ids: Iterable[int], reason: str) -> None:
        new = set(ids) - keep
        reasons[reason] += len(new)
        keep.update(new)

    add(tok.added_token_ids(), "special/added")
    add(tok.base_alphabet_ids(), "byte alphabet")
    add(cfg.extra_ids, "referenced by config")
    if corpus_counts:
        add((i for i, c in corpus_counts.items() if c >= cfg.min_freq), "seen in corpus")
    if cfg.keep_ascii:
        ascii_ids = []
        for token, i in tok.vocab.items():
            raw = token_bytes(token)
            if raw is not None and raw.isascii():
                ascii_ids.append(i)
        add(ascii_ids, "pure ASCII (code/English)")
    if cfg.scripts:
        pattern = _script_pattern(cfg.scripts)
        script_ids = []
        for token, i in tok.vocab.items():
            text = token_text(token)
            if text is not None and pattern.match(text):
                script_ids.append(i)
        add(script_ids, f"script: {'+'.join(cfg.scripts)}")

    closed = tok.ancestry_closure(keep)
    reasons["BPE merge ancestry"] += len(closed - keep)
    return Selection(keep=sorted(closed), reasons=reasons)


def script_histogram(tok: TokenizerJson) -> Counter:
    """How many vocabulary entries belong to each Unicode script."""
    patterns = {s: regex.compile(rf"\p{{Script={s}}}") for s in KNOWN_SCRIPTS}
    hist: Counter = Counter()
    for token in tok.vocab:
        text = token_text(token)
        if text is None:
            hist["(partial UTF-8 byte sequence)"] += 1
            continue
        found = [s for s, p in patterns.items() if p.search(text)]
        if not found:
            hist["Common (digits, punctuation, symbols)"] += 1
        elif len(found) == 1:
            hist[found[0]] += 1
        else:
            hist["(mixed)"] += 1
    return hist
