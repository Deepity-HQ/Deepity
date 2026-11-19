"""Block quantizers for GGUF writing: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 and the K-quants Q2_K to Q6_K.

Each function takes a float array whose element count is a multiple of the block size (32, or 256 for the
K-quants) and returns the raw little-endian blocks in ggml's layout. They do not copy llama.cpp's
quantizers step for step: scales are found by a short least-squares search, so a file is valid and decodes
within quantization error, but its bytes differ from llama.cpp's. IQ quants need codebooks and importance
data and are read-only.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

_SEARCH_STEPS = 19


def _f16(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Float16 bytes of `values` (shape [..., 1] kept as [..., 2]) and the float32 they decode to."""
    half = values.astype(np.float16)
    return half.view(np.uint8).reshape(*values.shape[:-1], -1), half.astype(np.float32)


def _inverse(scale: np.ndarray) -> np.ndarray:
    return np.divide(1.0, scale, out=np.zeros_like(scale), where=scale != 0)


def _round(x: np.ndarray) -> np.ndarray:
    return np.floor(x + 0.5)


def _blocks(x: np.ndarray, size: int) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float32).reshape(-1, size)


def _pack_nibbles(q: np.ndarray) -> np.ndarray:
    """32 values 0..15 per block to 16 bytes: the low nibble holds the first half, the high nibble the second."""
    return (q[:, :16] | (q[:, 16:] << 4)).astype(np.uint8)


def _join(*parts: np.ndarray) -> bytes:
    return np.concatenate([np.ascontiguousarray(p).reshape(p.shape[0], -1).view(np.uint8) for p in parts], axis=1).tobytes()


def _signed_max(blocks: np.ndarray) -> np.ndarray:
    return np.take_along_axis(blocks, np.abs(blocks).argmax(axis=1)[:, None], axis=1)


def quantize_q4_0(x: np.ndarray) -> bytes:
    blocks = _blocks(x, 32)
    d_bytes, d = _f16(_signed_max(blocks) / -8.0)
    q = np.clip(np.floor(blocks * _inverse(d) + 8.5), 0, 15).astype(np.uint8)
    return _join(d_bytes, _pack_nibbles(q))


def _fit_offset(blocks: np.ndarray, top: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scale, offset and the stored float16 bytes for blocks coded as x = scale * q + offset, q in 0..top."""
    lo, hi = blocks.min(axis=1, keepdims=True), blocks.max(axis=1, keepdims=True)
    d_bytes, d = _f16((hi - lo) / top)
    m_bytes, m = _f16(lo)
    return d_bytes, d, m_bytes, m


def quantize_q4_1(x: np.ndarray) -> bytes:
    blocks = _blocks(x, 32)
    d_bytes, d, m_bytes, m = _fit_offset(blocks, 15)
    q = np.clip(np.floor((blocks - m) * _inverse(d) + 0.5), 0, 15).astype(np.uint8)
    return _join(d_bytes, m_bytes, _pack_nibbles(q))


def _high_bits(q: np.ndarray) -> np.ndarray:
    """The fifth bit of 32 values as one little-endian uint32 per block, element j in bit j."""
    bits = ((q >> 4) & 1).astype(np.uint64) << np.arange(32, dtype=np.uint64)
    return bits.sum(axis=1).astype("<u4").view(np.uint8).reshape(-1, 4)


def quantize_q5_0(x: np.ndarray) -> bytes:
    blocks = _blocks(x, 32)
    d_bytes, d = _f16(_signed_max(blocks) / -16.0)
    q = np.clip(np.floor(blocks * _inverse(d) + 16.5), 0, 31).astype(np.uint8)
    return _join(d_bytes, _high_bits(q), _pack_nibbles(q & 15))


def quantize_q5_1(x: np.ndarray) -> bytes:
    blocks = _blocks(x, 32)
    d_bytes, d, m_bytes, m = _fit_offset(blocks, 31)
    q = np.clip(np.floor((blocks - m) * _inverse(d) + 0.5), 0, 31).astype(np.uint8)
    return _join(d_bytes, m_bytes, _high_bits(q), _pack_nibbles(q & 15))


def quantize_q8_0(x: np.ndarray) -> bytes:
    blocks = _blocks(x, 32)
    d_bytes, scale = _f16(np.abs(blocks).max(axis=1, keepdims=True) / 127.0)  # quantize against the stored scale
    scaled = blocks * _inverse(scale)
    q = np.clip(np.sign(scaled) * np.floor(np.abs(scaled) + 0.5), -127, 127).astype(np.int8)
    return _join(d_bytes, q)


# ---- K-quants -------------------------------------------------------------------------------------------------


def _fit_scale_and_offset(sub: np.ndarray, top: int) -> tuple[np.ndarray, np.ndarray]:
    """Per sub-block scale >= 0 and offset >= 0 for x = scale * q - offset, q in 0..top (sub is [..., n])."""
    n = sub.shape[-1]
    lo = np.minimum(sub.min(axis=-1, keepdims=True), 0.0)
    scale = (sub.max(axis=-1, keepdims=True) - lo) / top
    best, best_error = (scale, -lo), np.full(scale.shape, np.inf, dtype=np.float32)
    for _ in range(4):
        q = np.clip(_round((sub - lo) * _inverse(scale)), 0, top)
        error = ((scale * q + lo - sub) ** 2).sum(axis=-1, keepdims=True)
        better = error < best_error
        best = (np.where(better, scale, best[0]), np.where(better, -lo, best[1]))
        best_error = np.where(better, error, best_error)
        sq, sx, sqq, sqx = q.sum(-1, keepdims=True), sub.sum(-1, keepdims=True), (q * q).sum(-1, keepdims=True), (q * sub).sum(-1, keepdims=True)
        det = n * sqq - sq * sq
        ok = det > 1e-12
        safe = np.where(ok, det, 1.0)
        new_scale = np.where(ok, (n * sqx - sx * sq) / safe, scale)
        new_lo = np.where(ok, (sqq * sx - sq * sqx) / safe, lo)
        positive = new_lo > 0  # the offset may only lower the values; refit through zero
        new_scale = np.where(positive, np.divide(sqx, sqq, out=np.zeros_like(sqx), where=sqq > 0), new_scale)
        scale, lo = np.maximum(new_scale, 0.0), np.where(positive, 0.0, new_lo)
    return best


def _quantize_scales(values: np.ndarray, top: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Second-level quantization: float16 super-scale d and integer codes 0..top, as (d bytes, d, codes)."""
    d_bytes, d = _f16(values.max(axis=1, keepdims=True) / top)
    return d_bytes, d, np.clip(_round(values * _inverse(d)), 0, top).astype(np.uint8)


def _offset_quants(x: np.ndarray, sub_size: int, top: int, scale_top: int):
    """Quantize 256-element blocks as x = d * sc * q - dmin * m with per-sub-block integer sc and m."""
    sub = _blocks(x, 256).reshape(-1, 256 // sub_size, sub_size)
    scale, offset = _fit_scale_and_offset(sub, top)
    d_bytes, d, sc = _quantize_scales(scale[..., 0], scale_top)
    m_bytes, dmin, mc = _quantize_scales(offset[..., 0], scale_top)
    eff_scale, eff_offset = d[..., None] * sc[..., None], dmin[..., None] * mc[..., None]
    q = np.clip(_round((sub + eff_offset) * _inverse(eff_scale)), 0, top).astype(np.uint8)
    return d_bytes, m_bytes, sc, mc, q.reshape(-1, 256)


def _pack_scales_k4(sc: np.ndarray, mc: np.ndarray) -> np.ndarray:
    """Eight 6-bit scales and eight 6-bit minimums into the 12 bytes of Q4_K and Q5_K."""
    out = np.zeros((sc.shape[0], 12), dtype=np.uint8)
    out[:, :4] = sc[:, :4] | ((sc[:, 4:] >> 4) << 6)
    out[:, 4:8] = mc[:, :4] | ((mc[:, 4:] >> 4) << 6)
    out[:, 8:] = (sc[:, 4:] & 15) | ((mc[:, 4:] & 15) << 4)
    return out


def _pack_chunks(q: np.ndarray) -> np.ndarray:
    """256 values per block in four chunks of 64: the low nibble holds a chunk's first 32, the high nibble its last 32."""
    chunks = q.reshape(-1, 4, 2, 32)
    return ((chunks[:, :, 0] & 15) | ((chunks[:, :, 1] & 15) << 4)).reshape(-1, 128).astype(np.uint8)


def quantize_q4_k(x: np.ndarray) -> bytes:
    d, m, sc, mc, q = _offset_quants(x, 32, 15, 63)
    return _join(d, m, _pack_scales_k4(sc, mc), _pack_chunks(q))


def quantize_q5_k(x: np.ndarray) -> bytes:
    d, m, sc, mc, q = _offset_quants(x, 32, 31, 63)
    chunks = q.reshape(-1, 4, 2, 32)
    high = np.zeros((q.shape[0], 32), dtype=np.uint8)
    for c in range(4):
        high |= (((chunks[:, c, 0] >> 4) & 1) << (2 * c)).astype(np.uint8)
        high |= (((chunks[:, c, 1] >> 4) & 1) << (2 * c + 1)).astype(np.uint8)
    return _join(d, m, _pack_scales_k4(sc, mc), high, _pack_chunks(q))


def _pack_two_bit(q: np.ndarray) -> np.ndarray:
    """256 values 0..3: two groups of 128, each 4 shift planes of two 16-value halves, to 64 bytes."""
    planes = q.reshape(-1, 2, 4, 2, 16)
    out = np.zeros((q.shape[0], 2, 2, 16), dtype=np.uint8)
    for j in range(4):
        out |= (planes[:, :, j] << (2 * j)).astype(np.uint8)
    return out.reshape(-1, 64)


def quantize_q2_k(x: np.ndarray) -> bytes:
    d, m, sc, mc, q = _offset_quants(x, 16, 3, 15)
    return _join(sc | (mc << 4), _pack_two_bit(q), d, m)


def _symmetric_scales(sub: np.ndarray, nmax: int) -> np.ndarray:
    """Per sub-block scale s with x ~ s * l for integer l in -nmax..nmax-1, by trying nearby step sizes."""
    peak = _signed_max(sub.reshape(-1, sub.shape[-1])).reshape(sub.shape[:-1] + (1,))
    best = np.zeros(peak.shape, dtype=np.float32)
    best_score = np.zeros(peak.shape, dtype=np.float32)
    for step in range(-(_SEARCH_STEPS // 2), _SEARCH_STEPS // 2 + 1):
        inverse = -(nmax + 0.1 * step) * _inverse(peak)
        l = np.clip(_round(inverse * sub), -nmax, nmax - 1)
        sumlx, suml2 = (sub * l).sum(-1, keepdims=True), (l * l).sum(-1, keepdims=True)
        score = np.divide(sumlx * sumlx, suml2, out=np.zeros_like(sumlx), where=suml2 > 0)
        better = (suml2 > 0) & (score > best_score)
        best = np.where(better, np.divide(sumlx, suml2, out=np.zeros_like(sumlx), where=suml2 > 0), best)
        best_score = np.where(better, score, best_score)
    return best[..., 0]


def _symmetric_quants(x: np.ndarray, nmax: int, scale_top: int):
    """Quantize 256-element blocks as x = d * sc * l with 16 signed integer sc: (d bytes, sc, l)."""
    sub = _blocks(x, 256).reshape(-1, 16, 16)
    scales = _symmetric_scales(sub, nmax)
    biggest = np.take_along_axis(scales, np.abs(scales).argmax(axis=1)[:, None], axis=1)
    d_bytes, d = _f16(biggest / -scale_top)
    sc = np.clip(_round(scales * _inverse(d)), -scale_top, scale_top - 1).astype(np.int16)
    eff = (d * sc)[..., None]
    l = np.clip(_round(sub * _inverse(eff)), -nmax, nmax - 1).astype(np.int16)
    return d_bytes, sc, l.reshape(-1, 256)


def quantize_q3_k(x: np.ndarray) -> bytes:
    d_bytes, sc, l = _symmetric_quants(x, 4, 32)
    q = (l + 4).astype(np.uint8)  # 0..7: two low bits, one high bit
    planes = q.reshape(-1, 2, 4, 2, 16)
    mask = np.zeros((q.shape[0], 2, 16), dtype=np.uint8)
    for g in range(2):
        for j in range(4):
            mask |= (((planes[:, g, j] >> 2) & 1) << (4 * g + j)).astype(np.uint8)
    stored = (sc + 32).astype(np.uint8)  # 0..63
    scales = np.zeros((q.shape[0], 12), dtype=np.uint8)
    for s in range(16):
        low, high = stored[:, s] & 15, stored[:, s] >> 4
        quarter, slot = divmod(s, 4)
        scales[:, slot + 4 * (quarter % 2)] |= low << (4 * (quarter // 2))
        scales[:, 8 + slot] |= high << (2 * quarter)
    return _join(mask.reshape(-1, 32), _pack_two_bit(q & 3), scales, d_bytes)


def quantize_q6_k(x: np.ndarray) -> bytes:
    d_bytes, sc, l = _symmetric_quants(x, 32, 128)
    q = (l + 32).astype(np.uint8).reshape(-1, 2, 4, 32)
    ql = np.concatenate([(q[:, :, 0] & 15) | ((q[:, :, 2] & 15) << 4), (q[:, :, 1] & 15) | ((q[:, :, 3] & 15) << 4)], axis=2)
    qh = (q[:, :, 0] >> 4) | ((q[:, :, 1] >> 4) << 2) | ((q[:, :, 2] >> 4) << 4) | ((q[:, :, 3] >> 4) << 6)
    return _join(ql.reshape(-1, 128), qh.reshape(-1, 64), sc.astype(np.int8), d_bytes)


# name -> (ggml type id, elements per block, quantizer)
QUANTIZERS: dict[str, tuple[int, int, Callable[[np.ndarray], bytes]]] = {
    "q4_0": (2, 32, quantize_q4_0),
    "q4_1": (3, 32, quantize_q4_1),
    "q5_0": (6, 32, quantize_q5_0),
    "q5_1": (7, 32, quantize_q5_1),
    "q8_0": (8, 32, quantize_q8_0),
    "q2_k": (10, 256, quantize_q2_k),
    "q3_k": (11, 256, quantize_q3_k),
    "q4_k": (12, 256, quantize_q4_k),
    "q5_k": (13, 256, quantize_q5_k),
    "q6_k": (14, 256, quantize_q6_k),
}
