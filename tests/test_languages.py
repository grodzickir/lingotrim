import pytest

from lingotrim.corpus import parse_source
from lingotrim.languages import LANGUAGES, resolve


def test_every_table_entry_has_fineweb2_config_except_english():
    for code, lang in LANGUAGES.items():
        if code == "en":
            assert lang.fineweb2 is None
        else:
            iso3, script = lang.fineweb2.split("_")
            assert len(iso3) == 3 and script[0].isupper() and len(script) == 4


def test_resolve_accepts_iso1_and_raw_fineweb2_configs():
    assert resolve("PL").fineweb2 == "pol_Latn"
    assert resolve("ukr_cyrl").fineweb2 == "ukr_Cyrl"
    with pytest.raises(ValueError, match="unknown language"):
        resolve("xx")


def test_parse_source_forms():
    assert parse_source("pl").config == "pol_Latn"
    assert parse_source("en").dataset == "HuggingFaceFW/fineweb-edu"
    assert parse_source("code").column == "content"
    custom = parse_source("legal=my-org/legal-corpus:pl@body")
    assert (custom.name, custom.dataset, custom.config, custom.column) == (
        "legal", "my-org/legal-corpus", "pl", "body"
    )
    assert parse_source("x=org/ds").column == "text"
