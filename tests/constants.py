from pathlib import Path

from bitarray import bitarray

__all__ = ["e", "sha1"]


data_path = Path(__file__).parent / "data"

e = bitarray()
with open(data_path / "e.bin", "rb") as f:
    e.fromfile(f)
e = e[:1_000_000]  # convenience as only first million bits is used in examples

sha1 = bitarray()
with open(data_path / "sha1.bin", "rb") as f:
    sha1.fromfile(f)
