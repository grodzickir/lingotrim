from __future__ import annotations

import json

import numpy as np
from conftest import HIDDEN, TEXTS
from tokenizers import Tokenizer

from lingotrim.pipeline import trim_model
from lingotrim.safetensors_slice import read_header
from lingotrim.selection import SelectionConfig
from lingotrim.tokenizer_json import TokenizerJson
from lingotrim.verify import check_tokenization, old_to_new_ids


def load_tensor(path, name, np_dtype):
    start, infos, _ = read_header(path)
    info = next(t for t in infos if t.name == name)
    raw = path.read_bytes()[start + info.start : start + info.end]
    return np.frombuffer(raw, dtype=np_dtype).reshape(info.shape)


def test_target_languages_tokenize_identically(model_dir, corpus_dir, tmp_path):
    out = tmp_path / "out"
    report = trim_model(model_dir, out, [corpus_dir], SelectionConfig())

    assert report.vocab_after < report.vocab_before
    check = check_tokenization(model_dir, out, [TEXTS["en"], TEXTS["pl"], TEXTS["code"]])
    assert check.lossless, check.examples


def test_removed_languages_still_round_trip_through_bytes(model_dir, corpus_dir, tmp_path):
    out = tmp_path / "out"
    trim_model(model_dir, out, [corpus_dir], SelectionConfig())
    old = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    new = Tokenizer.from_file(str(out / "tokenizer.json"))

    for text in (TEXTS["ru"], TEXTS["zh"]):
        ids = new.encode(text, add_special_tokens=False).ids
        assert new.decode(ids) == text
        assert len(ids) > len(old.encode(text, add_special_tokens=False).ids)


def test_weight_rows_follow_the_id_remap(model_dir, corpus_dir, tmp_path):
    out = tmp_path / "out"
    report = trim_model(model_dir, out, [corpus_dir], SelectionConfig())
    remap = old_to_new_ids(model_dir, out)
    keep = sorted(remap, key=remap.get)

    shard1, shard2 = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
    for shard, name, dtype in ((shard1, "model.embed_tokens.weight", np.float32),
                               (shard2, "lm_head.weight", np.uint16),
                               (shard2, "lm_head.bias", np.float32)):
        old = load_tensor(model_dir / shard, name, dtype)
        new = load_tensor(out / shard, name, dtype)
        assert new.shape[0] == report.vocab_after
        np.testing.assert_array_equal(new[: len(keep)], old[keep])

    untouched_old = load_tensor(model_dir / shard1, "model.layers.0.w", np.float16)
    untouched_new = load_tensor(out / shard1, "model.layers.0.w", np.float16)
    np.testing.assert_array_equal(untouched_old, untouched_new)
    assert report.params_removed == (report.vocab_before - report.vocab_after) * (2 * HIDDEN + 1)


def test_configs_point_at_the_same_special_tokens(model_dir, corpus_dir, tmp_path):
    out = tmp_path / "out"
    report = trim_model(model_dir, out, [corpus_dir], SelectionConfig())
    tok = Tokenizer.from_file(str(out / "tokenizer.json"))
    config = json.loads((out / "config.json").read_text())
    gen = json.loads((out / "generation_config.json").read_text())
    tok_cfg = json.loads((out / "tokenizer_config.json").read_text())

    eos = tok.token_to_id("<|endoftext|>")
    assert config["eos_token_id"] == [eos] and gen["eos_token_id"] == eos
    assert config["text_config"]["image_token_id"] == tok.token_to_id("<|im_start|>")
    assert config["vocab_size"] == config["text_config"]["vocab_size"] == report.vocab_after
    assert tok_cfg["added_tokens_decoder"][str(eos)]["content"] == "<|endoftext|>"
    assert (out / "chat_template.jinja").exists()
    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] == report.weight_bytes_after


def test_every_kept_merge_has_kept_parents(model_dir, corpus_dir, tmp_path):
    out = tmp_path / "out"
    trim_model(model_dir, out, [corpus_dir], SelectionConfig())
    old = TokenizerJson.load(model_dir / "tokenizer.json")
    new = TokenizerJson.load(out / "tokenizer.json")

    producible = {a + b for a, b in new.merges}
    base = {old.id_to_token[i] for i in old.base_alphabet_ids()}
    base |= {t["content"] for t in new.added_tokens}
    for token in new.vocab:
        assert token in base or token in producible, token


def test_script_mode_keeps_whole_scripts(model_dir, tmp_path):
    out = tmp_path / "out"
    trim_model(model_dir, out, [], SelectionConfig(scripts=["Cyrillic"]))
    check = check_tokenization(model_dir, out, [TEXTS["ru"], TEXTS["en"]])
    assert check.lossless, check.examples
    assert check_tokenization(model_dir, out, [TEXTS["zh"]]).texts_with_removed_tokens == 1
