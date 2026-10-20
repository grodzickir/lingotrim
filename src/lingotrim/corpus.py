"""Download per-language text samples used to decide which tokens to keep."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .languages import resolve

FINEWEB2 = "HuggingFaceFW/fineweb-2"
ENGLISH = ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text")
CODE = ("codeparrot/codeparrot-clean-valid", None, "content")
HELDOUT_SHARE = 0.1


@dataclass(frozen=True)
class Source:
    name: str
    dataset: str
    config: str | None
    column: str


def parse_source(spec: str) -> Source:
    """`pl`, `code`, `pol_Latn` or `name=dataset[:config][@column]`."""
    if "=" in spec:
        name, rest = spec.split("=", 1)
        rest, _, column = rest.partition("@")
        dataset, _, config = rest.partition(":")
        return Source(name, dataset, config or None, column or "text")
    if spec == "code":
        return Source("code", *CODE)
    if spec == "en":
        return Source("en", *ENGLISH)
    lang = resolve(spec)
    if lang.fineweb2 is None:
        return Source(lang.code, *ENGLISH)
    return Source(lang.code, FINEWEB2, lang.fineweb2, "text")


def download(specs: list[str], out_dir: Path, chars_per_source: int = 20_000_000) -> list[Path]:
    """Write `<out>/fit/<name>.txt` and `<out>/heldout/<name>.txt`; skip sources already present."""
    from datasets import load_dataset

    fit_dir, heldout_dir = out_dir / "fit", out_dir / "heldout"
    fit_dir.mkdir(parents=True, exist_ok=True)
    heldout_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for spec in specs:
        src = parse_source(spec)
        fit_path, heldout_path = fit_dir / f"{src.name}.txt", heldout_dir / f"{src.name}.txt"
        if fit_path.exists() and heldout_path.exists():
            print(f"{src.name}: already downloaded")
            written.append(fit_path)
            continue
        ds = load_dataset(src.dataset, src.config, split="train", streaming=True)
        heldout_budget = int(chars_per_source * HELDOUT_SHARE)
        total = 0
        with heldout_path.open("w", encoding="utf-8", newline="\n") as heldout, \
                fit_path.open("w", encoding="utf-8", newline="\n") as fit:
            for row in ds:
                text = row[src.column].replace("\r", "") + "\n"
                (heldout if total < heldout_budget else fit).write(text)
                total += len(text)
                if total >= chars_per_source:
                    break
        written.append(fit_path)
        label = f"{src.dataset}{':' + src.config if src.config else ''}"
        print(f"{src.name}: {total:,} chars from {label}")
    return written
