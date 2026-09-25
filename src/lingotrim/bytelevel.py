from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def byte_encoder() -> dict[int, str]:
    """GPT-2 style byte -> printable unicode char mapping used by byte-level BPE."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs, strict=True)}


@lru_cache(maxsize=1)
def byte_decoder() -> dict[str, int]:
    return {c: b for b, c in byte_encoder().items()}


def token_bytes(token: str) -> bytes | None:
    """Raw bytes a byte-level BPE token stands for, or None if it is not byte-level."""
    dec = byte_decoder()
    try:
        return bytes(dec[ch] for ch in token)
    except KeyError:
        return None


def token_text(token: str) -> str | None:
    """Decoded text of a token, or None when it is a partial UTF-8 sequence."""
    raw = token_bytes(token)
    if raw is None:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
