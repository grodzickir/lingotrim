from pathlib import Path

from lingotrim.run import RunConfig


def test_sources_always_include_english_and_code_by_default():
    cfg = RunConfig(model="m", out=Path("o"), languages=["pl"])
    assert cfg.sources == ["en", "pl", "code"]
    assert cfg.tag == "pl-code"


def test_sources_keep_user_order_and_extras():
    cfg = RunConfig(model="m", out=Path("o"), languages=["en", "de"], code=False, extra_sources=["x=a/b"])
    assert cfg.sources == ["en", "de", "x=a/b"]
    assert cfg.tag == "en-de"
