from bisect import bisect_left
from collections import defaultdict
from functools import lru_cache
from itertools import product
from math import erfc, log, sqrt
from numbers import Real
from typing import (Iterable, Iterator, List, Literal, NamedTuple, Sequence,
                    Tuple, Union)

from bitarray import bitarray, frozenbitarray
from bitarray.util import ba2int
from scipy.fft import fft
from scipy.special import gammaincc
from scipy.stats import chisquare

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


BitArray = Sequence[Literal[0, 1]]


class Result(NamedTuple):
    statistic: Union[int, float]
    p: float


class ResultsMap(dict):
    @property
    def statistics(self) -> List[float]:
        return [result.statistic for result in self.values()]

    @property
    def pvalues(self) -> List[float]:
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


def _chunked(a: BitArray, blocksize: int, nblocks: int) -> Iterator[BitArray]:
    for i in range(0, blocksize * nblocks, blocksize):
        yield a[i:i + blocksize]


def block_frequency(bits, blocksize: int) -> Result:
    a = frozenbitarray(bits)
    n = len(a)
    nblocks = n // blocksize

    deviations = []
    for chunk in _chunked(a, blocksize, nblocks):
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


@lru_cache
def _binkey(okeys: Tuple[Real], key: Real) -> Real:
    i = bisect_left(okeys, key)
    left = okeys[i - 1]
    right = okeys[i]
    if abs(left - key) < abs(right - key):
        return left
    else:
        return right


def longest_runs(bits):
    a = frozenbitarray(bits)
    n = len(a)
    defaults = {
        # n: (blocksize, nblocks, intervals)
        128: (8, 16, (1, 2, 3, 4)),
        6272: (128, 49, (4, 5, 6, 7, 8, 9)),
        750000: (10 ** 4, 75, (10, 11, 12, 13, 14, 15, 16)),
    }
    try:
        key = min(k for k in defaults.keys() if k >= n)
    except ValueError as e:
        raise NotImplementedError(
            "Test implementation cannot handle sequences below length 128"
        ) from e
    blocksize, nblocks, intervals = defaults[key]

    maxlen_bins = {k: 0 for k in intervals}
    for chunk in _chunked(a, blocksize, nblocks):
        one_run_lengths = [len_ for val, len_ in _asruns(chunk) if val == 1]
        try:
            maxlen = max(one_run_lengths)
        except ValueError:
            maxlen = 0
        maxlen_bins[_binkey(intervals, maxlen)] += 1

    blocksize_probabilities = {
        # blocksize: <bin interval probabilities>
        8: [0.2148, 0.3672, 0.2305, 0.1875],
        128: [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124],
        512: [0.1170, 0.2460, 0.2523, 0.1755, 0.1027, 0.1124],
        1000: [0.1307, 0.2437, 0.2452, 0.1714, 0.1002, 0.1088],
        10000: [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727],
    }
    probabilities = blocksize_probabilities[blocksize]
    expected_bincounts = [prob * nblocks for prob in probabilities]

    chi2, p = chisquare(list(maxlen_bins.values()), expected_bincounts)

    return Result(chi2, p)


def _gf2_matrix_rank(matrix: Iterable[BitArray]) -> int:
    numbers = [ba2int(a) for a in matrix]

    rank = 0
    while len(numbers) > 0:
        pivot = numbers.pop()
        if pivot:
            rank += 1
            lsb = pivot & -pivot
            for i, num in enumerate(numbers):
                if lsb & num:
                    numbers[i] = num ^ pivot

    return rank


def matrix_rank(bits, matrix_dimen: Tuple[int, int]) -> Result:
    a = frozenbitarray(bits)
    n = len(a)
    nrows, ncols = matrix_dimen
    blocksize = nrows * ncols
    nblocks = n // blocksize

    ranks = []
    for chunk in _chunked(a, blocksize, nblocks):
        matrix = _chunked(chunk, ncols, nrows)
        rank = _gf2_matrix_rank(matrix)
        ranks.append(rank)

    fullrank = min(nrows, ncols)
    rankcounts = [0, 0, 0]
    for rank in ranks:
        if rank == fullrank:
            rankcounts[0] += 1
        elif rank == fullrank - 1:
            rankcounts[1] += 1
        else:
            rankcounts[2] += 1

    expected_rankcounts = (0.2888 * nblocks, 0.5776 * nblocks, 0.1336 * nblocks)

    chi2, p = chisquare(rankcounts, expected_rankcounts)

    return Result(chi2, p)


def spectral(bits) -> Result:
    a = bitarray(bits)
    n = len(a)
    if n % 2 != 0:
        a.pop()
    threshold = sqrt(log(1 / 0.05) * n)
    # TODO github.com/ilanschnell/bitarray/blob/master/examples/ndarray.py
    oscillations = [1 if b == 1 else -1 for b in a]

    fourier = fft(oscillations)
    half_fourier = fourier[: n // 2]
    peaks = [abs(n) for n in half_fourier]
    nbelow = sum(p < threshold for p in peaks)

    nbelow_expect = 0.95 * n / 2

    diff = nbelow - nbelow_expect
    normdiff = diff / sqrt((n * 0.95 * 0.05) / 4)
    p = erfc(abs(normdiff) / sqrt(2))

    return Result(normdiff, p)


def _windowed(a: BitArray, blocksize: int) -> Iterator[BitArray]:
    n = len(a)
    for i in range(0, n - blocksize + 1):
        yield a[i:i + blocksize]


def notm(bits, tempsize: int, blocksize: int) -> ResultsMap:
    a = frozenbitarray(bits)
    n = len(a)
    nblocks = n // blocksize

    block_counts = defaultdict(lambda: defaultdict(int))
    for i, chunk in enumerate(_chunked(a, blocksize, nblocks)):
        matches = defaultdict(int)
        for temp in _windowed(chunk, tempsize):
            matches[temp] += 1
        for temp, count in matches.items():
            block_counts[temp][i] = count

    count_expect = (blocksize - tempsize + 1) / 2 ** tempsize
    variance = blocksize * ((1 / 2 ** tempsize) - ((2 * tempsize - 1) / 2 ** (2 * tempsize)))
    results = ResultsMap()
    for temp in (frozenbitarray(p) for p in product([0, 1], repeat=tempsize)):
        count_diffs = [block_counts[temp][b] - count_expect for b in range(nblocks)]
        chi2 = sum(diff ** 2 / variance for diff in count_diffs)
        p = gammaincc(nblocks / 2, chi2 / 2)
        results[temp] = Result(chi2, p)

    return results


def otm(bits, **kwargs):
    pass


def universal(bits, **kwargs):
    pass


def complexity(bits, **kwargs):
    pass


def serial(bits, **kwargs):
    pass


def apen(bits, **kwargs):
    pass


def cusum(bits, **kwargs):
    pass


def excursions(bits, **kwargs):
    pass


def excursions_variant(bits, **kwargs):
    pass
