from math import erfc, sqrt
from typing import Iterator, List, Literal, NamedTuple, Sequence, Tuple, Union

from bitarray import frozenbitarray
from more_itertools import chunked
from scipy.special import gammaincc

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


BitArray = Sequence


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
    a = frozenbitarray(bits)

    n = len(a)
    ones = a.count(1)
    zeros = n - ones
    diff = abs(ones - zeros)
    normdiff = diff / sqrt(n)
    p = erfc(normdiff / sqrt(2))

    return Result(normdiff, p)


def block_frequency(bits, blocksize: int) -> Result:
    a = frozenbitarray(bits)

    n = len(a)
    nblocks = n // blocksize

    boundary = blocksize * nblocks
    deviations = []
    for chunk in chunked(a[:boundary], blocksize):
        ones = chunk.count(1)
        prop = ones / blocksize
        dev = prop - 1 / 2
        deviations.append(dev)

    chi2 = 4 * blocksize * sum(x ** 2 for x in deviations)
    p = gammaincc(nblocks / 2, chi2 / 2)

    return Result(chi2, p)


def _asruns(a: BitArray) -> Iterator[Tuple[Literal[0, 1], int]]:
    run_val = a[0]
    run_len = 1
    for value in a[1:]:
        if value == run_val:
            run_len += 1
        else:
            yield run_val, run_len
            run_val = value
            run_len = 1
    yield run_val, run_len


def runs(bits):
    a = frozenbitarray(bits)

    n = len(a)

    ones = a.count(1)
    prop_ones = ones / n
    prop_zeros = 1 - prop_ones

    nruns = sum(1 for _ in _asruns(a))
    p = erfc(
        abs(nruns - (2 * ones * prop_zeros)) /
        (2 * sqrt(2 * n) * prop_ones * prop_zeros)
    )

    return Result(nruns, p)


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
