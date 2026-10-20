from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

TEXTS = {
    "en": "The quick brown fox jumps over the lazy dog. Language models learn statistics "
    "of text and predict the next token. We keep only the languages we need. ",
    "pl": "Zażółć gęślą jaźń. Szybki brązowy lis przeskakuje nad leniwym psem. Modele "
    "językowe uczą się statystyki tekstu i przewidują następny token. Źdźbło trawy. ",
    "ru": "Съешь же ещё этих мягких французских булок, да выпей чаю. Языковые модели "
    "предсказывают следующий токен. ",
    "zh": "敏捷的棕色狐狸跳过了懒狗。语言模型学习文本的统计规律并预测下一个词元。",
    "code": "def trim(vocab: dict[str, int]) -> list[int]:\n    return sorted(vocab.values())\n"
    "for i in range(10):\n    print(f'value={i}')\n",
}
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
PAD_ROWS = 7
HIDDEN = 8


def write_safetensors(path: Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    header, blobs, offset = {}, [], 0
    for name, (dtype, arr) in tensors.items():
        raw = np.ascontiguousarray(arr).tobytes()
        header[name] = {"dtype": dtype, "shape": list(arr.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    hb = json.dumps(header).encode()
    hb += b" " * (-len(hb) % 8)
    path.write_bytes(struct.pack("<Q", len(hb)) + hb + b"".join(blobs))


def build_tokenizer(path: Path) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=900, special_tokens=SPECIALS, show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator([t * 20 for t in TEXTS.values()], trainer)
    tok.save(str(path))
    return tok


@pytest.fixture()
def model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "model"
    d.mkdir()
    tok = build_tokenizer(d / "tokenizer.json")
    vocab_rows = tok.get_vocab_size() + PAD_ROWS
    eos = tok.token_to_id("<|endoftext|>")
    image = tok.token_to_id("<|im_start|>")
    config = {"architectures": ["Fake"], "vocab_size": vocab_rows, "eos_token_id": [eos],
              "text_config": {"vocab_size": vocab_rows, "image_token_id": image}}
    (d / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (d / "generation_config.json").write_text(json.dumps({"eos_token_id": eos}), encoding="utf-8")
    (d / "tokenizer_config.json").write_text(json.dumps({
        "added_tokens_decoder": {str(tok.token_to_id(s)): {"content": s} for s in SPECIALS}
    }), encoding="utf-8")
    (d / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")

    rng = np.random.default_rng(0)
    embed = rng.standard_normal((vocab_rows, HIDDEN)).astype(np.float32)
    head_bf16 = rng.integers(0, 2**16, (vocab_rows, HIDDEN), dtype=np.uint16)
    layer = rng.standard_normal((HIDDEN, HIDDEN)).astype(np.float16)
    write_safetensors(d / "model-00001-of-00002.safetensors", {
        "model.embed_tokens.weight": ("F32", embed),
        "model.layers.0.w": ("F16", layer),
    })
    write_safetensors(d / "model-00002-of-00002.safetensors", {
        "lm_head.weight": ("BF16", head_bf16),
        "lm_head.bias": ("F32", embed[:, 0].copy()),
    })
    (d / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 0},
        "weight_map": {"model.embed_tokens.weight": "model-00001-of-00002.safetensors"},
    }), encoding="utf-8")
    return d


@pytest.fixture()
def corpus_dir(tmp_path: Path) -> Path:
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "en.txt").write_text(TEXTS["en"] * 3, encoding="utf-8")
    (d / "pl.txt").write_text(TEXTS["pl"] * 3, encoding="utf-8")
    return d
