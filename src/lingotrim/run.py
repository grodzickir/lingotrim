"""One-shot pipeline: model -> corpus -> trim -> verify -> GGUF -> quantize -> Ollama."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .selection import SelectionConfig, iter_corpus_texts

MODEL_FILES = ["*.safetensors", "*.json", "*.txt", "*.jinja", "LICENSE*", "*.md", "tokenizer.model"]


@dataclass
class RunConfig:
    model: str
    out: Path
    languages: list[str]
    code: bool = True
    corpus_chars: int = 20_000_000
    extra_sources: list[str] = field(default_factory=list)
    scripts: list[str] = field(default_factory=list)
    min_freq: int = 1
    quantize: list[str] = field(default_factory=list)
    gguf_type: str = "bf16"
    ollama: str | None = None
    llama_cpp: str | None = None
    verify_logits: bool = False

    @property
    def sources(self) -> list[str]:
        langs = list(dict.fromkeys(["en", *self.languages])) if "en" not in self.languages else self.languages
        return [*langs, *(["code"] if self.code else []), *self.extra_sources]

    @property
    def tag(self) -> str:
        return "-".join(self.languages) + ("-code" if self.code else "")


def resolve_model(spec: str, out: Path) -> Path:
    """A local directory, or a Hugging Face repo id downloaded into `<out>/original`."""
    if Path(spec).is_dir():
        return Path(spec)
    from huggingface_hub import snapshot_download

    target = out / "original"
    print(f"downloading {spec} -> {target}")
    snapshot_download(spec, local_dir=target, allow_patterns=MODEL_FILES)
    return target


def run(cfg: RunConfig) -> dict:
    from .corpus import download
    from .pipeline import trim_model
    from .verify import check_logits, check_tokenization

    cfg.out.mkdir(parents=True, exist_ok=True)
    result: dict = {"languages": cfg.languages, "code": cfg.code}

    print("== 1/6 model")
    original = resolve_model(cfg.model, cfg.out)
    result["original"] = str(original)

    print("== 2/6 corpus")
    corpus = cfg.out / "corpus"
    download(cfg.sources, corpus, cfg.corpus_chars)

    print("== 3/6 trim")
    trimmed = cfg.out / f"trimmed-{cfg.tag}"
    report_path = trimmed / "lingotrim_report.json"
    if report_path.exists():
        print(f"already trimmed: {trimmed}")
        result["trim"] = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = trim_model(
            original, trimmed, [corpus / "fit"],
            SelectionConfig(min_freq=cfg.min_freq, scripts=cfg.scripts),
        )
        result["trim"] = report.to_dict()
        print(f"vocab {report.vocab_before:,} -> {report.vocab_after:,} (-{report.vocab_reduction:.1%}), "
              f"{report.params_removed / 1e6:,.0f}M parameters removed")
    result["trimmed"] = str(trimmed)

    print("== 4/6 verify")
    texts = list(iter_corpus_texts([corpus / "heldout"], chunk_chars=500))
    check = check_tokenization(original, trimmed, texts)
    result["verify"] = {
        "lossless": check.lossless, "coverage": check.coverage, "texts": check.texts,
        "top_removed": check.top_removed,
    }
    print(f"tokenization {'LOSSLESS' if check.lossless else 'MISMATCH'}, "
          f"coverage {check.coverage:.4%} on {check.texts:,} held-out texts")
    if not check.lossless:
        raise RuntimeError("trimmed tokenizer diverged on kept tokens; aborting before export")
    if cfg.verify_logits:
        worst = check_logits(original, trimmed, texts[:8])
        result["verify"]["max_logit_diff"] = worst
        print(f"max logit diff on kept tokens: {worst:.2e}")

    if not cfg.quantize and not cfg.ollama and cfg.gguf_type is None:
        return _finish(cfg, result)

    print("== 5/6 gguf")
    from .llamacpp import LlamaCpp, ollama_create

    llama = LlamaCpp.ensure(cfg.llama_cpp)
    gguf_dir = cfg.out / "gguf"
    base = gguf_dir / f"{trimmed.name}-{cfg.gguf_type}.gguf"
    if not base.exists():
        pre = llama.detect_pre_tokenizer(original)
        print(f"pre-tokenizer: {pre}")
        llama.convert(trimmed, base, pre, cfg.gguf_type)
    result["gguf"] = {cfg.gguf_type: str(base)}

    print("== 6/6 quantize")
    final = base
    for qtype in cfg.quantize:
        dst = gguf_dir / f"{trimmed.name}-{qtype}.gguf"
        if not dst.exists():
            llama.quantize(base, dst, qtype)
        result["gguf"][qtype] = str(dst)
        final = dst
    if cfg.ollama:
        ollama_create(cfg.ollama, final)
        result["ollama"] = cfg.ollama
        print(f"ollama model: {cfg.ollama}")
    return _finish(cfg, result)


def _finish(cfg: RunConfig, result: dict) -> dict:
    path = cfg.out / f"run-{cfg.tag}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary: {path}")
    return result
