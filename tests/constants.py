from pathlib import Path

from bitarray import bitarray

__all__ = ["e"]

e = bitarray()
with open(Path(__file__).parent / "data" / "e.bin", "rb") as f:
    e.fromfile(f)
e = e[:1_000_000]
