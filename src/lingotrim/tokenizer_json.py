from __future__ import annotations

import copy
import json
from collections.abc import Iterable
from pathlib import Path

from .bytelevel import byte_encoder


class UnsupportedTokenizer(ValueError):
    pass


def _has_bytelevel(node) -> bool:
    if not isinstance(node, dict):
        return False
    if node.get("type") == "ByteLevel":
        return True
    return any(_has_bytelevel(child) for child in node.get("pretokenizers", []) or [])


class TokenizerJson:
    """Byte-level BPE `tokenizer.json` with the operations needed for vocabulary trimming."""

    def __init__(self, data: dict):
        model = data.get("model", {})
        if model.get("type") != "BPE":
            raise UnsupportedTokenizer(f"only BPE tokenizers are supported, got {model.get('type')}")
        if not _has_bytelevel(data.get("pre_tokenizer")):
            raise UnsupportedTokenizer(
                "only byte-level BPE tokenizers are supported (tested on Qwen)"
            )
        self.data = data
        self.vocab: dict[str, int] = model["vocab"]
        self.id_to_token = {i: t for t, i in self.vocab.items()}
        self._merges_as_strings = bool(model["merges"]) and isinstance(model["merges"][0], str)
        self.merges: list[tuple[str, str]] = [
            tuple(m.split(" ", 1)) if isinstance(m, str) else tuple(m) for m in model["merges"]
        ]
        self.added_tokens: list[dict] = data.get("added_tokens", [])

    @classmethod
    def load(cls, path: str | Path) -> TokenizerJson:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    @property
    def size(self) -> int:
        ids = list(self.vocab.values()) + [t["id"] for t in self.added_tokens]
        return max(ids) + 1

    def added_token_ids(self) -> set[int]:
        return {t["id"] for t in self.added_tokens}

    def base_alphabet_ids(self) -> set[int]:
        """The 256 single-byte tokens; keeping them preserves the no-UNK guarantee."""
        return {self.vocab[c] for c in byte_encoder().values() if c in self.vocab}

    def ancestry_closure(self, keep: Iterable[int]) -> set[int]:
        """Add every intermediate token needed by BPE merges to build the kept tokens."""
        parents: dict[str, tuple[str, str]] = {}
        for a, b in self.merges:
            parents.setdefault(a + b, (a, b))
        closed = set(keep)
        stack = [self.id_to_token[i] for i in closed if i in self.id_to_token]
        while stack:
            pair = parents.get(stack.pop())
            if not pair:
                continue
            for part in pair:
                pid = self.vocab.get(part)
                if pid is not None and pid not in closed:
                    closed.add(pid)
                    stack.append(part)
        return closed

    def trimmed(self, keep_sorted: list[int]) -> tuple[dict, dict[int, int]]:
        """Return (new tokenizer.json data, old_id -> new_id) keeping only `keep_sorted` ids."""
        remap = {old: new for new, old in enumerate(keep_sorted)}
        data = copy.deepcopy(self.data)
        model = data["model"]
        model["vocab"] = {t: remap[i] for t, i in self.vocab.items() if i in remap}
        kept_tokens = model["vocab"]
        merges = [
            (a, b)
            for a, b in self.merges
            if a in kept_tokens and b in kept_tokens and (a + b) in kept_tokens
        ]
        model["merges"] = [f"{a} {b}" for a, b in merges] if self._merges_as_strings else [
            [a, b] for a, b in merges
        ]
        for tok in data.get("added_tokens", []):
            tok["id"] = remap[tok["id"]]
        _remap_post_processor(data.get("post_processor"), remap)
        return data, remap


def _remap_post_processor(node, remap: dict[int, int]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "ids" and isinstance(value, list) and all(isinstance(v, int) for v in value):
                node[key] = [remap[v] for v in value]
            else:
                _remap_post_processor(value, remap)
    elif isinstance(node, list):
        for item in node:
            _remap_post_processor(item, remap)
