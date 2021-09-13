from bitarray import bitarray
from pytest import mark

from prngtest import _berlekamp_massey


@mark.parametrize(
    "a, min_size",
    [
        (bitarray("1101011110001"), 4),
        (bitarray("1001110110011101010010011"), 13),
    ],
)
def test_berlekamp_massey(a, min_size):
    assert _berlekamp_massey(a) == min_size
