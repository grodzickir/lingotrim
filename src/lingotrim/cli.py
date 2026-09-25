from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .selection import SelectionConfig, iter_corpus_texts, script_histogram
from .tokenizer_json import TokenizerJson


def _cmd_inspect(args) -> int:
    tok = TokenizerJson.load(Path(args.model) / "tokenizer.json")
    hist = script_histogram(tok)
    total = sum(hist.values())
    print(f"vocabulary: {len(tok.vocab):,} BPE tokens + {len(tok.added_tokens)} added tokens")
    for script, n in hist.most_common():
        print(f"  {script:<40} {n:>8,}  {n / total:6.1%}")
    return 0


def _cmd_corpus(args) -> int:
    from .corpus import download

    download(args.sources, Path(args.out), args.chars)
    return 0


def _cmd_trim(args) -> int:
    from .pipeline import trim_model

    cfg = SelectionConfig(
        min_freq=args.min_freq,
        keep_ascii=not args.no_ascii,
        scripts=[s for s in args.scripts.split(",") if s] if args.scripts else [],
    )
    report = trim_model(args.model, args.out, args.corpus, cfg)
    print(f"vocab: {report.vocab_before:,} -> {report.vocab_after:,} "
          f"(-{report.vocab_reduction:.1%})")
    for reason, n in report.reasons.items():
        print(f"  kept {n:>8,}  {reason}")
    if report.weight_bytes_before:
        gb = 1024**3
        print(f"weights: {report.weight_bytes_before / gb:.2f} GB -> "
              f"{report.weight_bytes_after / gb:.2f} GB, "
              f"{report.params_removed / 1e6:,.1f}M parameters removed")
    print(f"report: {Path(args.out) / 'lingotrim_report.json'}")
    return 0


def _cmd_verify(args) -> int:
    from .verify import check_logits, check_tokenization

    texts = list(iter_corpus_texts(args.corpus, chunk_chars=500))
    result = check_tokenization(Path(args.original), Path(args.trimmed), texts)
    status = "LOSSLESS" if result.lossless else "BUG: UNEXPECTED MISMATCH"
    print(f"tokenization: {status} on {result.texts:,} texts / {result.tokens:,} tokens")
    print(f"  vocabulary coverage: {result.coverage:.4%} of tokens kept "
          f"({result.texts_with_removed_tokens:,} texts used removed tokens, re-encoded as bytes)")
    if result.top_removed:
        print("  most frequent removed tokens: "
              + ", ".join(f"{t!r}×{n}" for t, n in result.top_removed))
    for example in result.examples:
        print(f"  differs: {example!r}")
    if args.logits:
        worst = check_logits(Path(args.original), Path(args.trimmed), texts[: args.logit_samples])
        print(f"logits: max |diff| on kept tokens = {worst:.2e}")
        if worst > args.atol:
            return 1
    if args.json:
        print(json.dumps(result.__dict__, ensure_ascii=False))
    return 0 if result.lossless else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lingotrim", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("inspect", help="show how the vocabulary splits across Unicode scripts")
    s.add_argument("model", help="model directory containing tokenizer.json")
    s.set_defaults(func=_cmd_inspect)

    s = sub.add_parser("corpus", help="download language samples (needs the `corpus` extra)")
    s.add_argument("sources", nargs="+", help="e.g. en pl code, or name=dataset[:config][@column]")
    s.add_argument("--out", default="corpus", help="output directory")
    s.add_argument("--chars", type=int, default=20_000_000, help="characters per source")
    s.set_defaults(func=_cmd_corpus)

    s = sub.add_parser("trim", help="trim tokenizer + embeddings + LM head")
    s.add_argument("model", help="Hugging Face model directory (safetensors)")
    s.add_argument("out", help="output directory (must be empty)")
    s.add_argument("--corpus", nargs="*", default=[], help="text files/dirs in target languages")
    s.add_argument("--min-freq", type=int, default=1, help="min corpus count to keep a token")
    s.add_argument("--scripts", default="", help="also keep all tokens of these Unicode scripts")
    s.add_argument("--no-ascii", action="store_true", help="do not force-keep pure ASCII tokens")
    s.set_defaults(func=_cmd_trim)

    s = sub.add_parser("verify", help="prove the trimmed model behaves identically")
    s.add_argument("original")
    s.add_argument("trimmed")
    s.add_argument("--corpus", nargs="+", required=True, help="held-out target-language text")
    s.add_argument("--logits", action="store_true", help="also compare logits (needs `verify`)")
    s.add_argument("--logit-samples", type=int, default=8)
    s.add_argument("--atol", type=float, default=1e-3)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=_cmd_verify)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
