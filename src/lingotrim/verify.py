from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from tokenizers import Tokenizer

from .tokenizer_json import TokenizerJson


@dataclass
class TokenizationCheck:
    texts: int = 0
    tokens: int = 0
    removed_token_hits: int = 0
    texts_with_removed_tokens: int = 0
    top_removed: list[tuple[str, int]] = field(default_factory=list)
    unexpected_mismatches: int = 0
    examples: list[str] = field(default_factory=list)

    @property
    def lossless(self) -> bool:
        """No text made only of kept tokens tokenizes differently."""
        return self.unexpected_mismatches == 0

    @property
    def coverage(self) -> float:
        return 1 - self.removed_token_hits / max(self.tokens, 1)


def old_to_new_ids(original: Path, trimmed: Path) -> dict[int, int]:
    old = TokenizerJson.load(original / "tokenizer.json")
    new = TokenizerJson.load(trimmed / "tokenizer.json")
    old_ids = {**old.vocab, **{t["content"]: t["id"] for t in old.added_tokens}}
    new_ids = {**new.vocab, **{t["content"]: t["id"] for t in new.added_tokens}}
    return {old_ids[t]: i for t, i in new_ids.items()}


def check_tokenization(original: Path, trimmed: Path, texts: Iterable[str]) -> TokenizationCheck:
    """Target-language text must tokenize to exactly the same tokens before and after."""
    remap = old_to_new_ids(original, trimmed)
    tok_old = Tokenizer.from_file(str(original / "tokenizer.json"))
    tok_new = Tokenizer.from_file(str(trimmed / "tokenizer.json"))
    result = TokenizationCheck()
    removed: Counter = Counter()
    for text in texts:
        old_ids = tok_old.encode(text, add_special_tokens=False).ids
        result.texts += 1
        result.tokens += len(old_ids)
        missing = [i for i in old_ids if i not in remap]
        if missing:
            removed.update(missing)
            result.removed_token_hits += len(missing)
            result.texts_with_removed_tokens += 1
            continue
        new_ids = tok_new.encode(text, add_special_tokens=False).ids
        if [remap[i] for i in old_ids] != new_ids:
            result.unexpected_mismatches += 1
            if len(result.examples) < 5:
                result.examples.append(text[:120])
    result.top_removed = [(tok_old.decode([i]), n) for i, n in removed.most_common(10)]
    return result


def check_logits(original: Path, trimmed: Path, texts: list[str], atol: float = 1e-3) -> float:
    """Max |logit difference| on kept tokens. Needs the `verify` extra; small models only."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    remap = old_to_new_ids(original, trimmed)
    kept_old = torch.tensor(sorted(remap, key=remap.get))
    tokenizer = AutoTokenizer.from_pretrained(trimmed)
    kwargs = {"torch_dtype": torch.float32}
    m_old = AutoModelForCausalLM.from_pretrained(original, **kwargs).eval()
    m_new = AutoModelForCausalLM.from_pretrained(trimmed, **kwargs).eval()
    inverse = {v: k for k, v in remap.items()}
    worst = 0.0
    with torch.no_grad():
        for text in texts:
            new_ids = tokenizer(text, return_tensors="pt").input_ids
            old_ids = new_ids.clone().apply_(lambda i: inverse[i])
            lo = m_old(old_ids).logits[..., : m_old.config.vocab_size][..., kept_old]
            ln = m_new(new_ids).logits[..., : len(kept_old)]
            worst = max(worst, (lo - ln).abs().max().item())
    return worst
