"""ZX Spectrum .scr screen file: 6144 bytes of bitmap in the interleaved screen order, then 768
attribute bytes (bit 6 bright, bits 5..3 paper, bits 2..0 ink; flash is never set).
"""
import numpy as np

H, W, BLOCK = 192, 256, 8

def _row_offsets():
    y = np.arange(H)
    return ((y & 0xC0) << 5) | ((y & 7) << 8) | ((y & 0x38) << 2)

def attr_bytes(idx_pairs: np.ndarray) -> np.ndarray:
    """(R, C, 2) palette indexes (paper, ink), 0..7 not bright and 8..15 bright, to attribute bytes."""
    paper, ink = idx_pairs[..., 0], idx_pairs[..., 1]
    bright = (paper >= 8) | (ink >= 8)
    return (bright * 0x40 | (paper & 7) << 3 | (ink & 7)).astype(np.uint8)

def to_scr(bitmap: np.ndarray, idx_pairs: np.ndarray) -> bytes:
    """bitmap (192, 256) truthy = ink; idx_pairs (24, 32, 2) as in attr_bytes."""
    assert bitmap.shape == (H, W) and idx_pairs.shape == (H // BLOCK, W // BLOCK, 2)
    rows = np.packbits(bitmap.astype(bool), axis=1)   # bit 7 is the leftmost pixel
    out = np.zeros(H * W // 8, dtype=np.uint8)
    for y, off in enumerate(_row_offsets()):
        out[off:off + W // 8] = rows[y]
    return out.tobytes() + attr_bytes(idx_pairs).tobytes()

if __name__ == '__main__':
    rng = np.random.default_rng(0)
    bitmap = rng.random((H, W)) < 0.5
    idx = rng.integers(0, 16, (H // BLOCK, W // BLOCK, 2))
    data = np.frombuffer(to_scr(bitmap, idx), dtype=np.uint8)
    assert data.size == 6912
    back = np.zeros((H, W // 8), dtype=np.uint8)
    for y, off in enumerate(_row_offsets()):
        back[y] = data[off:off + W // 8]
    assert (np.unpackbits(back, axis=1).astype(bool) == bitmap).all()
    attrs = data[6144:].reshape(H // BLOCK, W // BLOCK)
    assert ((attrs & 7) == (idx[..., 1] & 7)).all() and ((attrs >> 3 & 7) == (idx[..., 0] & 7)).all()
    assert ((attrs >> 6 & 1) == ((idx >= 8).any(-1))).all()
    assert _row_offsets()[1] == 256 and _row_offsets()[8] == 32 and _row_offsets()[64] == 2048
    print('ok')
