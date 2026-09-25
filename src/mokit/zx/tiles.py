"""Tile rungs of the pipeline: `Tileset` (K tiles of 8x8) and `TileMap` (cells x frames of tile indices + attrs).
See `docs/types.md`.

A tile packs into one uint64 (`pack_tiles`, row-major bits, native byte order: the same value AmaZX stores as
`tile_id`), so the empty tile is 0 and bitwise NOT is the inverted tile.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from mokit.graph import register_file_type
from mokit.types import Timing
from .screen import Patch, Screen

__all__ = ["Tileset", "TileMap", "pack_tiles", "unpack_tiles", "quantize"]


def pack_tiles(tiles: np.ndarray) -> np.ndarray:
    """(K, 8, 8) bool -> (K,) uint64."""
    tiles = np.ascontiguousarray(np.asarray(tiles, dtype=bool).reshape(-1, 8, 8))
    return np.frombuffer(np.packbits(tiles).tobytes(), dtype=np.uint64).copy()


def unpack_tiles(ids: np.ndarray) -> np.ndarray:
    """(K,) uint64 -> (K, 8, 8) bool."""
    ids = np.ascontiguousarray(np.asarray(ids, dtype=np.uint64))
    return np.unpackbits(np.frombuffer(ids.tobytes(), dtype=np.uint8)).reshape(-1, 8, 8).astype(bool)


@dataclass(frozen=True, eq=False)
class Tileset:
    """K tiles of 8x8 bool. No timing. File form: 8 bytes per tile, top row first, MSB left (a ZX font / charset)."""
    tiles: np.ndarray

    def __post_init__(self) -> None:
        tiles = np.asarray(self.tiles, dtype=bool)
        if tiles.ndim != 3 or tiles.shape[1:] != (8, 8):
            raise ValueError(f"Tileset.tiles must be (K,8,8), got {tiles.shape}")
        object.__setattr__(self, "tiles", tiles)

    def __len__(self) -> int:
        return len(self.tiles)

    @property
    def ids(self) -> np.ndarray:
        """(K,) uint64 packed tiles."""
        return pack_tiles(self.tiles)

    @property
    def empty_index(self) -> Optional[int]:
        """Index of the all-off tile, or None when the set has none (a font)."""
        hits = np.flatnonzero(~self.tiles.any(axis=(1, 2)))
        return int(hits[0]) if len(hits) else None

    @classmethod
    def from_ids(cls, ids) -> "Tileset":
        return cls(unpack_tiles(np.asarray(list(ids), dtype=np.uint64)))

    def to_bytes(self) -> bytes:
        return np.packbits(self.tiles, axis=-1).tobytes()

    @classmethod
    def from_bytes(cls, data: bytes) -> "Tileset":
        raw = np.frombuffer(data, dtype=np.uint8)
        if len(raw) % 8:
            raise ValueError(f"tileset bytes must be a multiple of 8, got {len(raw)}")
        return cls(np.unpackbits(raw.reshape(-1, 8, 1), axis=-1).astype(bool))

    @classmethod
    def load(cls, path) -> "Tileset":
        return cls.from_bytes(Path(path).read_bytes())

    def save(self, path) -> None:
        Path(path).write_bytes(self.to_bytes())


register_file_type(Tileset, ".bin", save=lambda v, p: v.save(p), load=Tileset.load)


@dataclass(frozen=True, eq=False)
class TileMap:
    """Cells x frames: `ids (N,R,C)` index into one `tileset` shared by the sequence, `attrs (N,R,C)` packed attr
    bytes, `Timing`. `from_patch` builds the identity tilemap (every distinct tile, the empty tile at index 0);
    `to_patch` / `to_screen` render it back."""
    ids: np.ndarray
    attrs: np.ndarray
    tileset: Tileset
    timing: Timing

    def __post_init__(self) -> None:
        ids, attrs = np.asarray(self.ids, dtype=np.int32), np.asarray(self.attrs, dtype=np.uint8)
        if ids.ndim != 3 or ids.shape != attrs.shape:
            raise ValueError(f"TileMap: ids {ids.shape} and attrs {attrs.shape} must both be (N,R,C)")
        if ids.size and (ids.min() < 0 or ids.max() >= len(self.tileset)):
            raise ValueError(f"TileMap: tile index out of range for a tileset of {len(self.tileset)}")
        if len(self.timing) != len(ids):
            raise ValueError(f"TileMap: {len(ids)} frames but timing has {len(self.timing)}")
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "attrs", attrs)

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def rows(self) -> int:
        return self.ids.shape[1]

    @property
    def cols(self) -> int:
        return self.ids.shape[2]

    @property
    def tile_ids(self) -> np.ndarray:
        """(N,R,C) uint64 packed tile per cell (AmaZX's `tile_id`; 0 = empty)."""
        return self.tileset.ids[self.ids]

    @classmethod
    def from_patch(cls, patch: Patch) -> "TileMap":
        n, rows, cols = patch.attrs.shape
        tiles = patch.pixels.reshape(n, rows, 8, cols, 8).transpose(0, 1, 3, 2, 4).reshape(-1, 8, 8)
        packed = pack_tiles(tiles)
        uniq, inverse = np.unique(np.concatenate([[np.uint64(0)], packed]), return_inverse=True)
        return cls(np.asarray(inverse[1:]).reshape(n, rows, cols), patch.attrs, Tileset(unpack_tiles(uniq)),
                   patch.timing)

    @classmethod
    def from_tile_ids(cls, tile_ids: np.ndarray, attrs: np.ndarray, timing: Timing) -> "TileMap":
        """From (N,R,C) packed uint64 tiles (AmaZX `tile_id`): the tileset is the distinct tiles, 0 first."""
        flat = np.asarray(tile_ids, dtype=np.uint64).reshape(-1)
        uniq, inverse = np.unique(np.concatenate([[np.uint64(0)], flat]), return_inverse=True)
        return cls(np.asarray(inverse[1:]).reshape(np.asarray(tile_ids).shape), attrs, Tileset(unpack_tiles(uniq)),
                   timing)

    def to_patch(self) -> Patch:
        n, rows, cols = self.ids.shape
        pixels = self.tileset.tiles[self.ids].transpose(0, 1, 3, 2, 4).reshape(n, rows * 8, cols * 8)
        return Patch(pixels, self.attrs, self.timing)

    def to_screen(self) -> Screen:
        p = self.to_patch()
        return Screen(p.pixels, p.attrs, p.timing)

    def with_tileset(self, tileset: Tileset, ids: np.ndarray) -> "TileMap":
        return dataclasses.replace(self, tileset=tileset, ids=ids)


def quantize(tilemap: TileMap, tileset: Tileset) -> TileMap:
    """Map every cell onto the nearest tile of `tileset` by Hamming distance (ties: lowest index). The identity
    map when the tileset already holds every tile used."""
    used = tilemap.tileset.ids
    target = tileset.ids
    lookup = {int(t): i for i, t in reversed(list(enumerate(target)))}     # lowest index wins
    mapping = np.empty(len(used), dtype=np.int32)
    src_bits = np.unpackbits(used.view(np.uint8).reshape(len(used), 8), axis=1)      # (K, 64)
    dst_bits = np.unpackbits(target.view(np.uint8).reshape(len(target), 8), axis=1)  # (M, 64)
    for k, tile in enumerate(used):
        hit = lookup.get(int(tile))
        if hit is None:
            hit = int(np.argmin((src_bits[k][np.newaxis] != dst_bits).sum(axis=1)))
        mapping[k] = hit
    return tilemap.with_tileset(tileset, mapping[tilemap.ids])
