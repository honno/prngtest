from math import erfc, sqrt
from typing import List, NamedTuple, Union

from bitarray import bitarray

__all__ = [
    "monobit",
    "block_frequency",
    "runs",
    "longest_runs",
    "matrix_rank",
    "spectral",
    "notm",
    "otm",
    "universal",
    "complexity",
    "serial",
    "apen",
    "cusum",
    "excursions",
    "excursions_variant",
]


class Result(NamedTuple):
    statistic: Union[int, float]
    p: float


class ResultsMap(dict):
    @property
    def statistic(self) -> List[float]:
        return [result.statistic for result in self.values()]

    @property
    def p(self) -> List[float]:
        return [result.p for result in self.values()]


def monobit(bits) -> Result:
    a = bitarray(bits)

    n = len(a)
    ones = a.count(1)
    zeros = n - ones
    diff = abs(ones - zeros)
    normdiff = diff / sqrt(n)
    p = erfc(normdiff / sqrt(2))

    return Result(normdiff, p)


def block_frequency():
    pass


def runs():
    pass


def longest_runs():
    pass


def matrix_rank():
    pass


def spectral():
    pass


def notm():
    pass


def otm():
    pass


def universal():
    pass


def complexity():
    pass


def serial():
    pass


def apen():
    pass


def cusum():
    pass


def excursions():
    pass


def excursions_variant():
    pass
