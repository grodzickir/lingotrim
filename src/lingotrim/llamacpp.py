"""llama.cpp integration: fetch converter + binaries, export GGUF, quantize, import into Ollama."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"
SOURCE_TARBALL = "https://github.com/ggml-org/llama.cpp/archive/refs/tags/{tag}.tar.gz"
BINARY_ASSET = "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{asset}"


def home() -> Path:
    if env := os.environ.get("LINGOTRIM_HOME"):
        return Path(env)
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "lingotrim"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "lingotrim"


def _binary_asset(tag: str) -> str:
    machine = platform.machine().lower()
    arch = "arm64" if machine in {"arm64", "aarch64"} else "x64"
    if sys.platform == "win32":
        return f"llama-{tag}-bin-win-cpu-{arch}.zip"
    if sys.platform == "darwin":
        return f"llama-{tag}-bin-macos-{arch}.tar.gz"
    return f"llama-{tag}-bin-ubuntu-{arch}.tar.gz"


def latest_tag() -> str:
    with urllib.request.urlopen(RELEASES_API, timeout=30) as resp:
        releases = json.load(resp)
    for rel in releases:
        if rel["tag_name"].startswith("b") and rel["tag_name"][1:].isdigit():
            return rel["tag_name"]
    raise RuntimeError("no llama.cpp build release found")


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as resp, dst.open("wb") as fh:
        shutil.copyfileobj(resp, fh)


def _extract(archive: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dst)
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(dst, filter="data")


# get_vocab_base_pre moved from ModelBase to TextModel in newer llama.cpp; patch wherever it lives.
_FIND_VOCAB_CLASS = (
    "import conversion.base as base\n"
    "cls = next(c for c in (getattr(base, 'TextModel', None), base.ModelBase)"
    " if c is not None and 'get_vocab_base_pre' in vars(c))\n"
)


class LlamaCpp:
    """A pinned llama.cpp checkout (for `convert_hf_to_gguf.py`) plus prebuilt binaries."""

    def __init__(self, source: Path, bin_dir: Path | None):
        self.source = source
        self.bin_dir = bin_dir

    @classmethod
    def ensure(cls, source: str | Path | None = None, tag: str | None = None) -> LlamaCpp:
        if source or (env := os.environ.get("LLAMA_CPP_DIR")):
            src = Path(source or env)
            return cls(src, _find_bin_dir(src))
        tag = tag or latest_tag()
        root = home() / "llama.cpp" / tag
        src = root / "src"
        if not (src / "convert_hf_to_gguf.py").exists():
            archive = root / "source.tar.gz"
            _download(SOURCE_TARBALL.format(tag=tag), archive)
            _extract(archive, root / "src-tmp")
            shutil.move(str(next((root / "src-tmp").iterdir())), str(src))
            shutil.rmtree(root / "src-tmp")
            archive.unlink()
        bin_root = root / "bin"
        if _find_bin_dir(bin_root) is None:
            asset = _binary_asset(tag)
            archive = root / asset
            _download(BINARY_ASSET.format(tag=tag, asset=asset), archive)
            _extract(archive, bin_root)
            archive.unlink()
        return cls(src, _find_bin_dir(bin_root))

    def binary(self, name: str) -> Path:
        exe = name + (".exe" if sys.platform == "win32" else "")
        if self.bin_dir and (self.bin_dir / exe).exists():
            return self.bin_dir / exe
        if found := shutil.which(name):
            return Path(found)
        raise FileNotFoundError(f"{exe} not found; set LLAMA_CPP_DIR or let lingotrim download a release")

    def detect_pre_tokenizer(self, original_model: Path) -> str:
        """Ask llama.cpp which `tokenizer.ggml.pre` the untrimmed tokenizer maps to."""
        code = (
            "import sys, logging; logging.disable(logging.CRITICAL); sys.path.insert(0, sys.argv[1])\n"
            "from transformers import AutoTokenizer\n"
            + _FIND_VOCAB_CLASS
            + "tok = AutoTokenizer.from_pretrained(sys.argv[2])\n"
            "print('PRE=' + cls.get_vocab_base_pre(None, tok))\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", code, str(self.source), str(original_model)],
            capture_output=True, text=True, check=True,
        ).stdout
        return next(line for line in out.splitlines() if line.startswith("PRE=")).removeprefix("PRE=")

    def convert(self, model_dir: Path, out_file: Path, pre_tokenizer: str, outtype: str = "bf16") -> Path:
        """Run convert_hf_to_gguf.py with the pre-tokenizer pinned (the trimmed vocab changes its hash)."""
        out_file.parent.mkdir(parents=True, exist_ok=True)
        code = (
            "import sys, runpy; sys.path.insert(0, sys.argv[1])\n"
            + _FIND_VOCAB_CLASS
            + "pre = sys.argv[2]\n"
            "cls.get_vocab_base_pre = lambda self, tokenizer: pre\n"
            "sys.argv = [sys.argv[1] + '/convert_hf_to_gguf.py'] + sys.argv[3:]\n"
            "runpy.run_path(sys.argv[0], run_name='__main__')\n"
        )
        args = [str(model_dir), "--outfile", str(out_file), "--outtype", outtype]
        subprocess.run(
            [sys.executable, "-c", code, str(self.source), pre_tokenizer, *args], check=True
        )
        return out_file

    def quantize(self, src: Path, dst: Path, qtype: str, imatrix: Path | None = None) -> Path:
        cmd = [str(self.binary("llama-quantize"))]
        if imatrix:
            cmd += ["--imatrix", str(imatrix)]
        cmd += [str(src), str(dst), qtype]
        subprocess.run(cmd, check=True)
        return dst


def _find_bin_dir(root: Path) -> Path | None:
    exe = "llama-quantize.exe" if sys.platform == "win32" else "llama-quantize"
    if not root.exists():
        return None
    for candidate in [root, root / "build" / "bin", *root.rglob("")]:
        if (candidate / exe).exists():
            return candidate
    return None


def ollama_create(name: str, gguf: Path, params: dict[str, str] | None = None) -> None:
    modelfile = gguf.with_suffix(".Modelfile")
    lines = [f"FROM {gguf.resolve()}"]
    for key, value in (params or {}).items():
        lines.append(f"PARAMETER {key} {value}")
    modelfile.write_text("\n".join(lines) + "\n", encoding="utf-8")
    subprocess.run(["ollama", "create", name, "-f", str(modelfile)], check=True)
