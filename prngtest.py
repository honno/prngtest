from bisect import bisect_left
from collections import defaultdict
from functools import lru_cache
from math import erfc, log, sqrt
from numbers import Real
from typing import (Dict, Iterable, Iterator, List, Literal, NamedTuple, Tuple,
                    Union)

import numpy as np
from bitarray import bitarray, frozenbitarray
from bitarray.util import ba2int, int2ba, zeros
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


class Result(NamedTuple):
    statistic: Union[int, float]
    p: float


class ResultsTuple(tuple):
    @property
    def statistics(self) -> List[float]:
        return [result.statistic for result in self]

    @property
    def pvalues(self) -> List[float]:
        return [result.p for result in self]


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


def _chunked(a: bitarray, nblocks: int, blocksize: int) -> Iterator[bitarray]:
    for i in range(0, blocksize * nblocks, blocksize):
        yield a[i:i + blocksize]


def block_frequency(bits, blocksize: int) -> Result:
    a = frozenbitarray(bits)
    n = len(a)
    nblocks = n // blocksize

    deviations = []
    for chunk in _chunked(a, nblocks, blocksize):
        ones = chunk.count(1)
        prop = ones / blocksize
        dev = prop - 1 / 2
        deviations.append(dev)

    chi2 = 4 * blocksize * sum(x ** 2 for x in deviations)
    p = gammaincc(nblocks / 2, chi2 / 2)

    return Result(chi2, p)


def _asruns(a: bitarray) -> Iterator[Tuple[Literal[0, 1], int]]:
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
    i = min(bisect_left(okeys, key), len(okeys) - 1)
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
        # n: (nblocks, blocksize, intervals)
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
    nblocks, blocksize, intervals = defaults[key]

    max_len_bins = {k: 0 for k in intervals}
    for chunk in _chunked(a, nblocks, blocksize):
        one_run_lengths = [len_ for val, len_ in _asruns(chunk) if val == 1]
        try:
            max_len = max(one_run_lengths)
        except ValueError:
            max_len = 0
        max_len_bins[_binkey(intervals, max_len)] += 1

    blocksize_probs = {
        # blocksize: <bin interval probs>
        8: [0.2148, 0.3672, 0.2305, 0.1875],
        128: [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124],
        512: [0.1170, 0.2460, 0.2523, 0.1755, 0.1027, 0.1124],
        1000: [0.1307, 0.2437, 0.2452, 0.1714, 0.1002, 0.1088],
        10000: [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727],
    }
    expected_bin_counts = [prob * nblocks for prob in blocksize_probs[blocksize]]

    chi2, p = chisquare(list(max_len_bins.values()), expected_bin_counts)

    return Result(chi2, p)


def _gf2_matrix_rank(matrix: Iterable[bitarray]) -> int:
    nums = [ba2int(a) for a in matrix]

    rank = 0
    while len(nums) > 0:
        pivot = nums.pop()
        if pivot:
            rank += 1
            lsb = pivot & -pivot
            for i, num in enumerate(nums):
                if lsb & num:
                    nums[i] = num ^ pivot

    return rank


def matrix_rank(bits, matrix_dimen: Tuple[int, int]) -> Result:
    a = frozenbitarray(bits)
    n = len(a)
    nrows, ncols = matrix_dimen
    blocksize = nrows * ncols
    nblocks = n // blocksize

    ranks = []
    for chunk in _chunked(a, nblocks, blocksize):
        matrix = _chunked(chunk, nrows, ncols)
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
    x = np.frombuffer(a.unpack(), dtype=np.bool_)
    oscillations = x.astype(np.int8)
    np.putmask(oscillations, ~x, np.int8(-1))

    fourier = fft(oscillations)
    half_fourier = fourier[: n // 2]
    peaks = [abs(n) for n in half_fourier]
    nbelow = sum(p < threshold for p in peaks)

    nbelow_expect = 0.95 * n / 2

    diff = nbelow - nbelow_expect
    normdiff = diff / sqrt((n * 0.95 * 0.05) / 4)
    p = erfc(abs(normdiff) / sqrt(2))

    return Result(normdiff, p)


def _windowed(a: bitarray, blocksize: int) -> Iterator[bitarray]:
    n = len(a)
    for i in range(0, n - blocksize + 1):
        yield a[i:i + blocksize]


def _product(length: int) -> Iterator[frozenbitarray]:
    for n in range(2 ** length):
        yield frozenbitarray(int2ba(n, length=length))


def notm(bits, tempsize: int, blocksize: int) -> Dict[bitarray, Result]:
    a = frozenbitarray(bits)
    n = len(a)
    nblocks = n // blocksize

    block_counts = defaultdict(lambda: defaultdict(int))
    for i, chunk in enumerate(_chunked(a, nblocks, blocksize)):
        matches = defaultdict(int)
        for window in _windowed(chunk, tempsize):
            matches[window] += 1
        for temp, count in matches.items():
            block_counts[temp][i] = count

    count_expect = (blocksize - tempsize + 1) / 2 ** tempsize
    variance = blocksize * ((1 / 2 ** tempsize) - ((2 * tempsize - 1) / 2 ** (2 * tempsize)))
    results = ResultsMap()
    for temp in _product(tempsize):
        count_diffs = [block_counts[temp][b] - count_expect for b in range(nblocks)]
        chi2 = sum(diff ** 2 / variance for diff in count_diffs)
        p = gammaincc(nblocks / 2, chi2 / 2)
        results[temp] = Result(chi2, p)

    return results


def otm(bits, tempsize: int, blocksize: int) -> Result:
    a = frozenbitarray(bits)
    n = len(a)
    nblocks = n // blocksize
    a = a[: nblocks * blocksize]
    temp = ~zeros(tempsize)

    block_matches = []
    for chunk in _chunked(a, nblocks, blocksize):
        matches = sum(window == temp for window in _windowed(chunk, tempsize))
        block_matches.append(matches)
    tallies = [0, 0, 0, 0, 0, 0]
    for matches in block_matches:
        tallies[min(matches, 5)] += 1

    tally_dist = [0.367879, 0.183939, 0.137954, 0.099634, 0.069935, 0.140656]
    expected_tallies = [prob * nblocks for prob in tally_dist]

    chi2, p = chisquare(tallies, expected_tallies)

    return Result(chi2, p)


def universal(bits, blocksize: int, init_nblocks: int) -> Result:
    a = frozenbitarray(bits)
    n = len(a)
    # defaults = {
    #     # n: (blocksize, init_nblocks)
    #     387840: (6, 640),
    #     904960: (7, 1280),
    #     2068480: (8, 2560),
    #     4654080: (9, 5120),
    #     10342400: (10, 10240),
    #     22753280: (11, 20480),
    #     49643520: (12, 40960),
    #     107560960: (13, 81920),
    #     231669760: (14, 163840),
    #     496435200: (15, 327680),
    #     1059061760: (16, 655360),
    # }
    nblocks = n // blocksize
    test_nblocks = nblocks - init_nblocks
    a = a[: nblocks * blocksize]
    bnd = init_nblocks * blocksize
    init, test = a[:bnd], a[bnd:]

    last_temp_indices = defaultdict(int)
    for i, temp in enumerate(_chunked(init, init_nblocks, blocksize), 1):
        last_temp_indices[temp] = i

    log2_gaps_acc = 0
    for i, temp in enumerate(_chunked(test, test_nblocks, blocksize), init_nblocks + 1):
        gap = i - last_temp_indices[temp]
        log2_gaps_acc += log(gap, 2)
        last_temp_indices[temp] = i

    blocksize_probs = {
        # blocksize: (mean, variance)
        1: (0.7326495, 0.690),
        2: (1.5374383, 1.338),
        3: (2.4016068, 1.901),
        4: (3.3112247, 2.358),
        5: (4.2534266, 2.705),
        6: (5.2177052, 2.954),
        7: (6.1962507, 3.125),
        8: (7.1836656, 3.238),
        9: (8.1764248, 3.311),
        10: (9.1723243, 3.356),
        11: (10.170032, 3.384),
        12: (11.168765, 3.401),
        13: (12.168070, 3.410),
        14: (13.167693, 3.416),
        15: (14.167488, 3.419),
        16: (15.167379, 3.421),
    }
    mean_expect, variance = blocksize_probs[blocksize]

    norm_gaps = log2_gaps_acc / test_nblocks
    p = erfc(abs((norm_gaps - mean_expect) / (sqrt(2 * variance))))

    return Result(norm_gaps, p)


def _berlekamp_massey(a: bitarray) -> int:
    n = len(a)
    errloc = zeros(n)
    errloc[0] = 1
    min_size = 0
    nloops = -1
    errlock_prev = errloc.copy()

    for i, bit in enumerate(a):
        discrepancy = bit
        for bit1, bit2 in zip(a[i - min_size:i][::-1], errloc[1: min_size + 1]):
            product = bit1 & bit2
            discrepancy = discrepancy ^ product
        if discrepancy:
            errloc_temp = errloc.copy()
            recalc = bitarray(
                bit1 ^ bit2 for bit1, bit2 in zip(errloc[i - nloops: n], errlock_prev)
            )
            errloc[i - nloops: n] = recalc
            if min_size <= i / 2:
                min_size = i + 1 - min_size
                nloops = i
                errlock_prev = errloc_temp

    return min_size


def complexity(bits, blocksize: int) -> Result:
    a = frozenbitarray(bits)
    n = len(a)
    nblocks = n // blocksize
    a = a[: nblocks * blocksize]
    mean_expect = (
        blocksize / 2 +
        (9 + (-(1 ** (blocksize + 1)))) / 36 -
        (blocksize / 3 + 2 / 9) / 2 ** blocksize
    )

    intervals = (-3, -2, -1, 0, 1, 2, 3)
    variance_bins = {k: 0 for k in [-3, -2, -1, 0, 1, 2, 3]}
    for chunk in _chunked(a, nblocks, blocksize):
        lin_complex = _berlekamp_massey(chunk)
        variance = (-1) ** blocksize * (lin_complex - mean_expect) + 2 / 9
        variance_bins[_binkey(intervals, variance)] += 1

    probs = [0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]
    expected_bin_counts = [nblocks * prob for prob in probs]

    chi2, p = chisquare(list(variance_bins.values()), expected_bin_counts)

    return Result(chi2, p)


def serial(bits, blocksize) -> Tuple[Result, Result]:
    a = frozenbitarray(bits)
    n = len(a)

    norm_sums = {}
    for tempsize in [blocksize, blocksize - 1, blocksize - 2]:
        if tempsize > 0:
            ouroboros = a + a[: tempsize - 1]
            counts = defaultdict(int)
            for window in _windowed(ouroboros, tempsize):
                counts[window] += 1
            sum_squares = sum(count ** 2 for count in counts.values())
            norm_sums[tempsize] = (2 ** tempsize / n) * sum_squares - n
        else:
            norm_sums[tempsize]

    norm_sum_delta1 = norm_sums[blocksize] - norm_sums[blocksize - 1]
    p1 = gammaincc(2 ** (blocksize - 2), norm_sum_delta1 / 2)
    normsum_delta2 = norm_sums[blocksize] - 2 * norm_sums[blocksize - 1] + norm_sums[blocksize - 2]
    p2 = gammaincc(2 ** (blocksize - 3), normsum_delta2 / 2)

    return ResultsTuple(
        (Result(norm_sum_delta1, p1), Result(normsum_delta2, p2))
    )


def apen(bits, blocksize: int) -> Result:
    a = frozenbitarray(bits)
    n = len(a)

    phis = []
    for tempsize in [blocksize, blocksize + 1]:
        ouroboros = a + a[: tempsize - 1]
        temp_counts = defaultdict(int)
        for window in _windowed(ouroboros, tempsize):
            temp_counts[window] += 1
        norm_counts = [count / n for count in temp_counts.values()]
        phi = sum(count * log(count) for count in norm_counts)
        phis.append(phi)

    apen = phis[0] - phis[1]
    chi2 = 2 * n * (log(2) - apen)
    p = gammaincc(2 ** (blocksize - 1), chi2 / 2)

    return Result(chi2, p)


def cusum(bits, **kwargs):
    pass


def excursions(bits, **kwargs):
    pass


def excursions_variant(bits, **kwargs):
    pass
