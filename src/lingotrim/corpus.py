"""Download small per-language text samples used to decide which tokens to keep."""

from __future__ import annotations

from pathlib import Path

FINEWEB2 = "HuggingFaceFW/fineweb-2"
SOURCES: dict[str, tuple[str, str | None, str]] = {
    "en": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
    "pl": (FINEWEB2, "pol_Latn", "text"),
    "de": (FINEWEB2, "deu_Latn", "text"),
    "fr": (FINEWEB2, "fra_Latn", "text"),
    "es": (FINEWEB2, "spa_Latn", "text"),
    "it": (FINEWEB2, "ita_Latn", "text"),
    "pt": (FINEWEB2, "por_Latn", "text"),
    "nl": (FINEWEB2, "nld_Latn", "text"),
    "cs": (FINEWEB2, "ces_Latn", "text"),
    "uk": (FINEWEB2, "ukr_Cyrl", "text"),
    "code": ("codeparrot/codeparrot-clean-valid", None, "content"),
}


def parse_source(spec: str) -> tuple[str, str, str | None, str]:
    """`pl` or `name=dataset[:config][@column]` -> (name, dataset, config, column)."""
    if "=" not in spec:
        if spec not in SOURCES:
            raise ValueError(f"unknown language {spec!r}; use name=dataset[:config][@column]")
        return (spec, *SOURCES[spec])
    name, rest = spec.split("=", 1)
    rest, _, column = rest.partition("@")
    dataset, _, config = rest.partition(":")
    return name, dataset, config or None, column or "text"


def download(specs: list[str], out_dir: Path, chars_per_source: int = 20_000_000) -> list[Path]:
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for spec in specs:
        name, dataset, config, column = parse_source(spec)
        ds = load_dataset(dataset, config, split="train", streaming=True)
        path = out_dir / f"{name}.txt"
        total = 0
        with path.open("w", encoding="utf-8") as fh:
            for row in ds:
                text = row[column]
                fh.write(text.replace("\r", "") + "\n")
                total += len(text)
                if total >= chars_per_source:
                    break
        written.append(path)
        print(f"{name}: {total:,} chars from {dataset}{':' + config if config else ''} -> {path}")
    return written
