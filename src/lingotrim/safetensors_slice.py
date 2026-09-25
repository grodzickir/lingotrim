"""Dtype-agnostic, streaming row slicing of safetensors files (no torch required)."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DTYPE_SIZES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
    "U32": 4, "I32": 4, "F32": 4,
    "U64": 8, "I64": 8, "F64": 8,
}
_VIEW = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}
_COPY_CHUNK = 64 * 1024 * 1024


@dataclass
class TensorInfo:
    name: str
    dtype: str
    shape: list[int]
    start: int
    end: int


def read_header(path: Path) -> tuple[int, list[TensorInfo], dict | None]:
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", None)
    infos = [
        TensorInfo(k, v["dtype"], v["shape"], *v["data_offsets"]) for k, v in header.items()
    ]
    infos.sort(key=lambda t: t.start)
    return 8 + n, infos, meta


def vocab_axis(info: TensorInfo, vocab_rows: int) -> int | None:
    axes = [a for a, d in enumerate(info.shape) if d == vocab_rows]
    if len(axes) > 1:
        raise ValueError(f"{info.name}: ambiguous vocab axis in shape {info.shape}")
    return axes[0] if axes else None


def slice_file(src: Path, dst: Path, vocab_rows: int, keep: np.ndarray) -> dict[str, list[int]]:
    """Copy `src` to `dst`, keeping only `keep` indices along every vocab-sized axis.

    Returns {tensor name: new shape} for the tensors that were sliced.
    """
    data_start, infos, meta = read_header(src)
    plan = []
    sliced: dict[str, list[int]] = {}
    offset = 0
    for info in infos:
        axis = vocab_axis(info, vocab_rows)
        shape = list(info.shape)
        if axis is not None:
            if info.dtype not in DTYPE_SIZES:
                raise ValueError(f"{info.name}: unsupported dtype {info.dtype} for slicing")
            shape[axis] = len(keep)
            sliced[info.name] = shape
            nbytes = int(np.prod(shape, dtype=np.int64)) * DTYPE_SIZES[info.dtype]
        else:
            nbytes = info.end - info.start
        plan.append((info, axis, shape, offset, offset + nbytes))
        offset += nbytes

    header: dict = {}
    if meta is not None:
        header["__metadata__"] = meta
    for info, _, shape, start, end in plan:
        header[info.name] = {"dtype": info.dtype, "shape": shape, "data_offsets": [start, end]}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * (-len(header_bytes) % 8)

    with src.open("rb") as fin, dst.open("wb") as fout:
        fout.write(struct.pack("<Q", len(header_bytes)))
        fout.write(header_bytes)
        for info, axis, _, _, _ in plan:
            if axis is None:
                fin.seek(data_start + info.start)
                remaining = info.end - info.start
                while remaining:
                    chunk = fin.read(min(_COPY_CHUNK, remaining))
                    fout.write(chunk)
                    remaining -= len(chunk)
            else:
                view = np.memmap(
                    src,
                    dtype=_VIEW[DTYPE_SIZES[info.dtype]],
                    mode="r",
                    offset=data_start + info.start,
                    shape=tuple(info.shape),
                )
                fout.write(np.ascontiguousarray(np.take(view, keep, axis=axis)).tobytes())
                del view
    return sliced


def total_size(path: Path) -> int:
    _, infos, _ = read_header(path)
    return sum(t.end - t.start for t in infos)
