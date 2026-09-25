from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from .safetensors_slice import read_header, slice_file
from .selection import SelectionConfig, count_corpus_tokens, iter_corpus_texts, select_tokens
from .tokenizer_json import TokenizerJson

TOKEN_ID_KEY = re.compile(r"(^|_)token_ids?$")
REWRITTEN = {
    "tokenizer.json", "tokenizer_config.json", "config.json", "generation_config.json",
    "vocab.json", "merges.txt", "model.safetensors.index.json",
}


@dataclass
class TrimReport:
    source: str
    vocab_before: int
    vocab_after: int
    reasons: dict[str, int]
    weight_bytes_before: int = 0
    weight_bytes_after: int = 0
    params_removed: int = 0
    sliced_tensors: dict[str, list[int]] = field(default_factory=dict)

    @property
    def vocab_reduction(self) -> float:
        return 1 - self.vocab_after / self.vocab_before

    def to_dict(self) -> dict:
        return {**asdict(self), "vocab_reduction": round(self.vocab_reduction, 4)}


def _load_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _dump_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _walk_token_ids(node, fn) -> None:
    """Apply fn(value) -> new value to every `*_token_id(s)` field, recursively."""
    if isinstance(node, dict):
        for key, value in node.items():
            if TOKEN_ID_KEY.search(key) and value is not None:
                node[key] = fn(value)
            else:
                _walk_token_ids(value, fn)
    elif isinstance(node, list):
        for item in node:
            _walk_token_ids(item, fn)


def _referenced_ids(*configs: dict | None) -> set[int]:
    found: set[int] = set()

    def collect(value):
        found.update(v for v in (value if isinstance(value, list) else [value]) if isinstance(v, int))
        return value

    for cfg in configs:
        if cfg:
            _walk_token_ids(cfg, collect)
    return found


def _vocab_rows(config: dict | None, fallback: int) -> int:
    if not config:
        return fallback
    for scope in (config, config.get("text_config") or {}):
        if isinstance(scope.get("vocab_size"), int):
            return scope["vocab_size"]
    return fallback


def _set_vocab_size(node, old: int, new: int) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "vocab_size" and value == old:
                node[key] = new
            else:
                _set_vocab_size(value, old, new)


def trim_model(
    src: str | Path,
    dst: str | Path,
    corpus: list[str | Path],
    cfg: SelectionConfig | None = None,
) -> TrimReport:
    src, dst = Path(src), Path(dst)
    cfg = cfg or SelectionConfig()
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f"output directory {dst} is not empty")
    dst.mkdir(parents=True, exist_ok=True)

    tok = TokenizerJson.load(src / "tokenizer.json")
    config = _load_json(src / "config.json")
    gen_config = _load_json(src / "generation_config.json")
    vocab_rows = _vocab_rows(config, tok.size)
    if vocab_rows < tok.size:
        raise ValueError(f"config vocab_size {vocab_rows} < tokenizer size {tok.size}")

    counts: Counter | None = None
    if corpus:
        fast = Tokenizer.from_file(str(src / "tokenizer.json"))
        counts = count_corpus_tokens(fast, iter_corpus_texts(corpus))
    cfg.extra_ids |= {i for i in _referenced_ids(config, gen_config) if 0 <= i < vocab_rows}
    selection = select_tokens(tok, counts, cfg)
    keep = selection.keep

    new_tok, remap = tok.trimmed(keep)
    _dump_json(dst / "tokenizer.json", new_tok)
    _write_slow_tokenizer_files(src, dst, new_tok)
    _write_tokenizer_config(src, dst, remap)
    for name, data in (("config.json", config), ("generation_config.json", gen_config)):
        if data is not None:
            _walk_token_ids(data, lambda v: [remap[i] for i in v] if isinstance(v, list) else remap[v])
            _set_vocab_size(data, vocab_rows, len(keep))
            _dump_json(dst / name, data)

    report = TrimReport(
        source=str(src),
        vocab_before=vocab_rows,
        vocab_after=len(keep),
        reasons=dict(selection.reasons),
    )
    _trim_weights(src, dst, vocab_rows, np.asarray(keep, dtype=np.int64), report)

    for f in src.iterdir():
        if f.is_file() and f.name not in REWRITTEN and f.suffix != ".safetensors":
            if f.suffix in {".bin", ".pt", ".pth", ".gguf"}:
                continue
            shutil.copy2(f, dst / f.name)
    _dump_json(dst / "lingotrim_report.json", report.to_dict())
    return report


def _write_tokenizer_config(src: Path, dst: Path, remap: dict[int, int]) -> None:
    data = _load_json(src / "tokenizer_config.json")
    if data is None:
        return
    if "added_tokens_decoder" in data:
        data["added_tokens_decoder"] = {
            str(remap[int(k)]): v for k, v in data["added_tokens_decoder"].items()
        }
    _dump_json(dst / "tokenizer_config.json", data)


def _write_slow_tokenizer_files(src: Path, dst: Path, new_tok: dict) -> None:
    if (src / "vocab.json").exists():
        _dump_json(dst / "vocab.json", new_tok["model"]["vocab"])
    if (src / "merges.txt").exists():
        merges = [m if isinstance(m, str) else " ".join(m) for m in new_tok["model"]["merges"]]
        (dst / "merges.txt").write_text("#version: 0.2\n" + "\n".join(merges) + "\n", encoding="utf-8")


def _trim_weights(src: Path, dst: Path, vocab_rows: int, keep: np.ndarray, report: TrimReport):
    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        return
    for shard in shards:
        _, infos, _ = read_header(shard)
        report.weight_bytes_before += sum(t.end - t.start for t in infos)
        sliced = slice_file(shard, dst / shard.name, vocab_rows, keep)
        report.sliced_tensors.update(sliced)
        _, new_infos, _ = read_header(dst / shard.name)
        report.weight_bytes_after += sum(t.end - t.start for t in new_infos)
        old_shapes = {t.name: t.shape for t in infos}
        for name, shape in sliced.items():
            report.params_removed += int(np.prod(old_shapes[name])) - int(np.prod(shape))
    index = _load_json(src / "model.safetensors.index.json")
    if index is not None:
        index.setdefault("metadata", {})["total_size"] = report.weight_bytes_after
        _dump_json(dst / "model.safetensors.index.json", index)
    if not report.sliced_tensors:
        raise ValueError(f"no tensor has a dimension equal to vocab_size={vocab_rows}")
