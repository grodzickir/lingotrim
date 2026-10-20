from pathlib import Path

import pytest

from lingotrim import llamacpp
from lingotrim.cli import build_parser
from lingotrim.llamacpp import LlamaCpp


@pytest.fixture
def calls(monkeypatch):
    recorded: list[list[str]] = []
    monkeypatch.setattr(llamacpp.subprocess, "run", lambda cmd, **kw: recorded.append(list(cmd)))
    return recorded


def _llama(monkeypatch, tmp_path) -> LlamaCpp:
    llama = LlamaCpp(tmp_path, None)
    monkeypatch.setattr(llama, "binary", lambda name: Path("bin") / name)
    return llama


def test_imatrix_command(monkeypatch, tmp_path, calls):
    out = _llama(monkeypatch, tmp_path).imatrix(
        tmp_path / "m.gguf", tmp_path / "calib.txt", tmp_path / "sub" / "m.imatrix", ctx_size=256
    )
    assert out == tmp_path / "sub" / "m.imatrix"
    assert out.parent.is_dir()
    (cmd,) = calls
    assert Path(cmd[0]).name == "llama-imatrix"
    flags = dict(zip(cmd[1::2], cmd[2::2], strict=False))
    assert flags["-m"] == str(tmp_path / "m.gguf")
    assert flags["-f"] == str(tmp_path / "calib.txt")
    assert flags["-o"] == str(out)
    assert flags["-c"] == "256"
    assert "--chunk" not in cmd  # `--chunk` means "start from chunk N" in llama-imatrix


def test_quantize_uses_imatrix_only_when_given(monkeypatch, tmp_path, calls):
    llama = _llama(monkeypatch, tmp_path)
    llama.quantize(tmp_path / "a.gguf", tmp_path / "b.gguf", "Q4_K_M")
    llama.quantize(tmp_path / "a.gguf", tmp_path / "c.gguf", "Q4_K_M", tmp_path / "m.imatrix")
    plain, with_imatrix = calls
    assert "--imatrix" not in plain
    assert plain[-3:] == [str(tmp_path / "a.gguf"), str(tmp_path / "b.gguf"), "Q4_K_M"]
    i = with_imatrix.index("--imatrix")
    assert with_imatrix[i + 1] == str(tmp_path / "m.imatrix")
    assert with_imatrix[-1] == "Q4_K_M"


def test_cli_imatrix_flags_parse():
    p = build_parser()
    assert p.parse_args(["imatrix", "m.gguf", "--corpus", "c.txt", "--out", "m.imatrix"]).ctx_size == 512
    assert p.parse_args(["export", "o", "t", "--out", "x.gguf", "--imatrix", "c.txt"]).imatrix == "c.txt"
    assert p.parse_args(["run", "m", "-l", "en", "--imatrix"]).imatrix is True


def test_binary_resolution_and_missing_error(monkeypatch, tmp_path):
    exe = llamacpp._exe("llama-quantize")
    (tmp_path / "build" / "bin").mkdir(parents=True)
    (tmp_path / "build" / "bin" / exe).write_text("", encoding="utf-8")
    assert llamacpp._find_bin_dir(tmp_path) == tmp_path / "build" / "bin"
    assert llamacpp._find_bin_dir(tmp_path / "missing") is None

    llama = LlamaCpp(tmp_path, llamacpp._find_bin_dir(tmp_path))
    assert llama.binary("llama-quantize") == tmp_path / "build" / "bin" / exe

    monkeypatch.setattr(llamacpp.shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="LLAMA_CPP_DIR"):
        llama.binary("llama-imatrix")


def test_binary_falls_back_to_path(monkeypatch, tmp_path):
    monkeypatch.setattr(llamacpp.shutil, "which", lambda name: str(tmp_path / name))
    assert LlamaCpp(tmp_path, None).binary("llama-imatrix") == tmp_path / "llama-imatrix"


def test_ollama_missing_gives_clear_error(monkeypatch, tmp_path):
    monkeypatch.setattr(llamacpp.shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="ollama"):
        llamacpp.ollama_create("x", tmp_path / "m.gguf")
    assert (tmp_path / "m.Modelfile").read_bytes().endswith(b"\n")
    assert b"\r" not in (tmp_path / "m.Modelfile").read_bytes()


def test_convert_requires_converter_script(tmp_path):
    with pytest.raises(FileNotFoundError, match="convert_hf_to_gguf.py"):
        LlamaCpp(tmp_path, None).convert(tmp_path, tmp_path / "o.gguf", "qwen2")
