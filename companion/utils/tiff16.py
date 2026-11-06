"""
tiff16.py
Lossless 16-bit RGB TIFF read/write for Lightroom round trips (stdlib + numpy/cv2 only).

Pillow decodes 48-bit RGB TIFFs to 8-bit and cannot write them, so the companion
reads the original 16-bit pixels with cv2 and writes the result with this
minimal baseline TIFF writer (little-endian, deflate strips, chunky RGB).
"""

import os
import struct
import tempfile
import zlib
from fractions import Fraction
from typing import Optional, Tuple

import cv2
import numpy as np

_SHORT, _LONG, _RATIONAL, _UNDEFINED = 3, 4, 5, 7
_ROWS_PER_STRIP = 64


def read_rgb16(path: str) -> Optional[np.ndarray]:
    """Return the file as a uint16 H x W x 3 RGB array, or None if it is not 16-bit colour."""
    # np.fromfile + imdecode instead of imread so non-ASCII Windows paths work
    arr = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None or arr.dtype != np.uint16 or arr.ndim != 3 or arr.shape[2] not in (3, 4):
        return None
    return np.ascontiguousarray(arr[:, :, 2::-1])  # BGR(A) -> RGB, alpha dropped


def _entry(tag: int, typ: int, values) -> Tuple[int, int, int, bytes]:
    if typ == _UNDEFINED:
        data = bytes(values)
        return tag, typ, len(data), data
    if typ == _RATIONAL:
        data = b"".join(struct.pack("<II", n, d) for n, d in values)
        return tag, typ, len(values), data
    fmt = "<%d%s" % (len(values), "H" if typ == _SHORT else "I")
    return tag, typ, len(values), struct.pack(fmt, *values)


def _rational(x: float) -> Tuple[int, int]:
    f = Fraction(float(x)).limit_denominator(10000)
    return f.numerator, f.denominator


def write_tiff16(path: str, rgb16: np.ndarray, icc_profile: Optional[bytes] = None,
                 dpi: Optional[Tuple[float, float]] = None) -> None:
    """
    Write a uint16 H x W x 3 array as a deflate-compressed 48-bit RGB TIFF.

    Embeds the ICC profile (tag 34675) and DPI (282/283/296) when given. The file is
    written to a temp file in the target directory and moved into place with
    os.replace, so a failed write never leaves a partial file at `path`.
    """
    if rgb16.dtype != np.uint16 or rgb16.ndim != 3 or rgb16.shape[2] != 3:
        raise ValueError(f"expected uint16 HxWx3 array, got {rgb16.dtype} {rgb16.shape}")
    h, w = rgb16.shape[:2]
    out_dir = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(suffix=".tif.part", dir=out_dir)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(b"II*\x00\x00\x00\x00\x00")  # IFD offset patched at the end
            offsets, counts = [], []
            for y in range(0, h, _ROWS_PER_STRIP):
                strip = zlib.compress(rgb16[y:y + _ROWS_PER_STRIP].astype("<u2", copy=False).tobytes(), 6)
                offsets.append(f.tell())
                counts.append(len(strip))
                f.write(strip)

            entries = [
                _entry(256, _LONG, [w]),
                _entry(257, _LONG, [h]),
                _entry(258, _SHORT, [16, 16, 16]),
                _entry(259, _SHORT, [8]),  # Adobe deflate
                _entry(262, _SHORT, [2]),  # RGB
                _entry(273, _LONG, offsets),
                _entry(277, _SHORT, [3]),
                _entry(278, _LONG, [_ROWS_PER_STRIP]),
                _entry(279, _LONG, counts),
                _entry(284, _SHORT, [1]),  # chunky
            ]
            if dpi:
                entries += [
                    _entry(282, _RATIONAL, [_rational(dpi[0])]),
                    _entry(283, _RATIONAL, [_rational(dpi[1])]),
                    _entry(296, _SHORT, [2]),  # inch
                ]
            if icc_profile:
                entries.append(_entry(34675, _UNDEFINED, icc_profile))
            entries.sort(key=lambda e: e[0])

            # Out-of-line values first (word aligned), then the IFD itself
            value_fields = []
            for tag, typ, count, data in entries:
                if len(data) <= 4:
                    value_fields.append(data.ljust(4, b"\x00"))
                    continue
                if f.tell() % 2:
                    f.write(b"\x00")
                value_fields.append(struct.pack("<I", f.tell()))
                f.write(data)
            if f.tell() % 2:
                f.write(b"\x00")
            ifd_offset = f.tell()
            if ifd_offset + 6 + 12 * len(entries) > 0xFFFFFFFF:
                raise ValueError("image too large for a classic TIFF (4 GB)")
            f.write(struct.pack("<H", len(entries)))
            for (tag, typ, count, _), field in zip(entries, value_fields):
                f.write(struct.pack("<HHI", tag, typ, count) + field)
            f.write(struct.pack("<I", 0))
            f.seek(4)
            f.write(struct.pack("<I", ifd_offset))
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def merge_into_16bit(base16: np.ndarray, base8, out8) -> np.ndarray:
    """
    Return a copy of `base16` with the 8-bit edit applied.

    Per channel: result = clip(base16 + (out8 - base8) * 257, 0, 65535). Where the 8-bit
    output equals the 8-bit base the delta is 0, so those values stay bit-identical.
    `base8` / `out8` are PIL RGB images of the same size; work is done in row strips
    so the only full-frame array allocated is the uint16 result.
    """
    h, w = base16.shape[:2]
    if base8.size != (w, h) or out8.size != (w, h):
        raise ValueError(f"size mismatch: 16-bit {w}x{h}, base {base8.size}, output {out8.size}")
    if out8.mode != "RGB":
        out8 = out8.convert("RGB")
    result = base16.copy()
    step = 256
    for y in range(0, h, step):
        box = (0, y, w, min(y + step, h))
        delta = np.asarray(out8.crop(box), np.int32) - np.asarray(base8.crop(box), np.int32)
        if delta.any():
            sl = result[box[1]:box[3]]
            sl[:] = np.clip(sl.astype(np.int32) + delta * 257, 0, 65535)
    return result
