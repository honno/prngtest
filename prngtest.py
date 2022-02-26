from bisect import bisect_left
from collections import defaultdict
from functools import lru_cache
from math import ceil, erfc, floor, log, sqrt
from numbers import Real
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)
from warnings import warn

import numpy as np
from bitarray import bitarray, frozenbitarray
from bitarray.util import ba2int, int2ba, zeros
from scipy.fft import fft
from scipy.special import gammaincc
from scipy.stats import chisquare, norm

__all__ = [
    "monobit",
    "blockfreq",
    "runs",
    "blockruns",
    "matrix",
    "spectral",
    "notm",
    "otm",
    "universal",
    "complexity",
    "serial",
    "apen",
    "cumsum",
    "excursions",
    "vexcursions",
]


# ------------------------------------------------------------------------------
# Helpers


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


def _check_bits_size(n: int, min_n: int):
    if n < min_n:
        raise ValueError(f"{n=} bits below required minimum of {min_n} bits")


def _check_recommendations(*args: Tuple[bool, str, str]):
    fails = [(f_vars, rec) for pass_, f_vars, rec in args if not pass_]
    if len(fails) > 0:
        lines = [f"{f_vars}, but NIST recommends {rec}" for f_vars, rec in fails]
        warn("\n".join(lines))


def _check_mutual_kwargs(
    arg1: Optional[Any], name1: str, arg2: Optional[Any], name2: str
):
    if (arg1 is None) ^ (arg2 is None):
        passed = name1 if arg2 is None else name2
        raise NotImplementedError(
            f"{name1}={arg1} and {name2}={arg2}, "
            f"but passing only {passed} is not supported"
        )


def _chunked(a: bitarray, nblocks: int, blocksize: int) -> Iterator[bitarray]:
    for i in range(0, nblocks * blocksize, blocksize):
        yield a[i : i + blocksize]


def _windowed(a: bitarray, blocksize: int) -> Iterator[bitarray]:
    n = len(a)
    for i in range(0, n - blocksize + 1):
        yield a[i : i + blocksize]


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


def _ascycles(x: np.ndarray) -> Iterator[np.ndarray]:
    split_at = np.arange(x.size)[x == 0]
    yield from np.split(x, split_at)


def _product(length: int) -> Iterator[frozenbitarray]:
    for n in range(2 ** length):
        yield frozenbitarray(int2ba(n, length=length))


@lru_cache
def _binkey(okeys: Tuple[Real], key: Real) -> Real:
    i = min(bisect_left(okeys, key), len(okeys) - 1)
    left = okeys[i - 1]
    right = okeys[i]
    if abs(left - key) < abs(right - key):
        return left
    else:
        return right


def _oscillate(a: bitarray) -> np.ndarray:
    x = np.frombuffer(a.unpack(), dtype=np.bool_)
    o = x.astype(np.int8)
    np.putmask(o, ~x, np.int8(-1))
    return o


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


def _berlekamp_massey(a: bitarray) -> int:
    n = len(a)
    errloc = zeros(n)
    errloc[0] = 1
    min_size = 0
    nloops = -1
    errlock_prev = errloc.copy()

    for i, bit in enumerate(a):
        discrepancy = bit
        for bit1, bit2 in zip(a[i - min_size : i][::-1], errloc[1 : min_size + 1]):
            product = bit1 & bit2
            discrepancy = discrepancy ^ product
        if discrepancy:
            errloc_temp = errloc.copy()
            recalc = bitarray(
                bit1 ^ bit2 for bit1, bit2 in zip(errloc[i - nloops : n], errlock_prev)
            )
            errloc[i - nloops : n] = recalc
            if min_size <= i / 2:
                min_size = i + 1 - min_size
                nloops = i
                errlock_prev = errloc_temp

    return min_size


# ------------------------------------------------------------------------------
# Randomness tests


Bits = Union[str, Iterable[int], bitarray]


def monobit(bits: Bits) -> Result:
    """Proportion of zeros and ones is compared to expected 1:1 ratio

    Parameters
    ----------
    bits : ``str | Iterable[int] | bitarray``
        Input sequence. If a ``str``, must contain only ``"0"`` or ``"1"``. If
        an iterable of integers, must contain only ``0`` or ``1``.

    Returns
    -------
    ``tuple[int | float, float]``
        The test's statistic and p-value.
    """
    a = frozenbitarray(bits)
    n = len(a)
    _check_bits_size(n, 2)
    _check_recommendations((n >= 100, f"{n=}", "n ≥ 100"))

    ones = a.count(1)
    zeros = n - ones
    diff = abs(ones - zeros)
    normdiff = diff / sqrt(n)
    p = erfc(normdiff / sqrt(2))

    return Result(normdiff, p)


def blockfreq(bits: Bits, blocksize: Optional[int] = None) -> Result:
    """Proportion of values per block is compared to expected 1:1 ratio

    Parameters
    ----------
    bits : ``str | Iterable[int] | bitarray``
        Input sequence. If a ``str``, must contain only ``"0"`` or ``"1"``. If
        an iterable of integers, must contain only ``0`` or ``1``.
    blocksize : ``Optional[int]``
        Size of the blocks that partition the given sequence. If ``None``, a
        suitable default will be found.

    Returns
    -------
    ``tuple[int | float, float]``
        The test's statistic and p-value.
    """
    a = frozenbitarray(bits)
    n = len(a)
    _check_bits_size(n, 8)
    if blocksize is None:
        blocksize = max(ceil(0.0125 * n), 4)
    nblocks = n // blocksize
    _check_recommendations(
        (n >= 100, f"{n=}", "n ≥ 100"),
        (blocksize >= 20, f"{blocksize=}", "blocksize ≥ 20"),
        (blocksize > 0.01 * n, f"{blocksize=}", "blocksize > 0.01 * n"),
        (nblocks < 100, f"{nblocks=}", "nblocks < 100"),
    )

    deviations = []
    for chunk in _chunked(a, nblocks, blocksize):
        ones = chunk.count(1)
        prop = ones / blocksize
        dev = prop - 1 / 2
        deviations.append(dev)

    chi2 = 4 * blocksize * sum(x ** 2 for x in deviations)
    p = gammaincc(nblocks / 2, chi2 / 2)

    return Result(chi2, p)


def runs(bits: Bits):
    """Number of runs is compared to expected result

    Parameters
    ----------
    bits : ``str | Iterable[int] | bitarray``
        Input sequence. If a ``str``, must contain only ``"0"`` or ``"1"``. If
        an iterable of integers, must contain only ``0`` or ``1``.

    Returns
    -------
    ``tuple[int | float, float]``
        The test's statistic and p-value.
    """
    a = frozenbitarray(bits)
    n = len(a)
    _check_bits_size(n, 2)
    _check_recommendations((n >= 100, f"{n=}", "n ≥ 100"))

    ones = a.count(1)
    prop_ones = ones / n
    prop_zeros = 1 - prop_ones
    if prop_zeros == 0 or prop_ones == 0:
        raise NotImplementedError("bits must contain both zeros and ones")

    nruns = sum(1 for _ in _asruns(a))
    p = erfc(
        abs(nruns - (2 * ones * prop_zeros))
        / (2 * sqrt(2 * n) * prop_ones * prop_zeros)
    )

    return Result(nruns, p)


def blockruns(bits: Bits):
    """Longest runs per block is compared to expected result

    Parameters
    ----------
    bits : ``str | Iterable[int] | bitarray``
        Input sequence. If a ``str``, must contain only ``"0"`` or ``"1"``. If
        an iterable of integers, must contain only ``0`` or ``1``.

    Returns
    -------
    ``tuple[int | float, float]``
        The test's statistic and p-value.
    """
    a = frozenbitarray(bits)
    n = len(a)
    # TODO: expose args
    _check_bits_size(n, 128)
    defaults = {
        # n: (blocksize, nblocks, intervals)
        128: (16, 8, (1, 2, 3, 4)),
        6272: (49, 128, (4, 5, 6, 7, 8, 9)),
        750000: (75, 10 ** 4, (10, 11, 12, 13, 14, 15, 16)),
    }
    key = max(k for k in defaults.keys() if k <= n)
    nblocks, blocksize, intervals = defaults[key]

    max_len_bins = {k: 0 for k in intervals}
    for chunk in _chunked(a[: nblocks * blocksize], nblocks, blocksize):
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
    key = max(k for k in blocksize_probs.keys() if k <= blocksize)
    expected_bin_counts = [prob * nblocks for prob in blocksize_probs[key]]

    chi2, p = chisquare(list(max_len_bins.values()), expected_bin_counts)

    return Result(chi2, p)


def matrix(
    bits: Bits, nrows: Optional[int] = None, ncols: Optional[int] = None
) -> Result:
    """Independence of neighbouring subsequences is compared to expected result

    Parameters
    ----------
    bits : ``str | Iterable[int] | bitarray``
        Input sequence. If a ``str``, must contain only ``"0"`` or ``"1"``. If
        an iterable of integers, must contain only ``0`` or ``1``.
    nrows : ``Optional[int]``
        Number of rows in each matrix. If ``None``, a suitable default will be
        found.
    ncols : ``Optional[int]``
        Number of columns in each matrix. If ``None``, a suitable default will
        be found.

    Returns
    -------
    ``tuple[int | float, float]``
        The test's statistic and p-value.
    """
    _check_mutual_kwargs(nrows, "nrows", ncols, "ncols")
    a = frozenbitarray(bits)
    n = len(a)
    _check_bits_size(n, 4)
    if nrows is None and ncols is None:
        blocksize = max(min(n // 38, 1024), 4)
        nrows = floor(sqrt(blocksize))
        ncols = blocksize // nrows
    blocksize = nrows * ncols
    nblocks = n // blocksize
    _check_recommendations(
        (n >= 128, f"{n=}", "n ≥ 128"),
        (blocksize >= 20, f"{ncols=} and {nrows=}", "nrows * ncols ≥ 20"),
    )

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


def spectral(bits: Bits) -> Result:
    """Potency of periodic features in sequence is compared to expected result

    Parameters
    ----------
    bits : ``str | Iterable[int] | bitarray``
        Input sequence. If a ``str``, must contain only ``"0"`` or ``"1"``. If
        an iterable of integers, must contain only ``0`` or ``1``.

    Returns
    -------
    ``tuple[int | float, float]``
        The test's statistic and p-value.
    """
    a = bitarray(bits)
    n = len(a)
    _check_bits_size(n, 2)
    _check_recommendations((n >= 1000, f"{n=}", "n ≥ 1000"))
    if n % 2 != 0:
        a.pop()
        n -= 1
    threshold = sqrt(log(1 / 0.05) * n)

    o = _oscillate(a)
    fourier = fft(o)
    half_fourier = fourier[: n // 2]
    peaks = [abs(n) for n in half_fourier]
    nbelow = sum(p < threshold for p in peaks)

    nbelow_expect = 0.95 * n / 2

    diff = nbelow - nbelow_expect
    normdiff = diff / sqrt((n * 0.95 * 0.05) / 4)
    p = erfc(abs(normdiff) / sqrt(2))

    return Result(normdiff, p)


def notm(
    bits, tempsize: Optional[int] = None, blocksize: Optional[int] = None
) -> Dict[bitarray, Result]:
    """Matches to template per block is compared to expected result

    Parameters
    ----------
    bits : ``str | Iterable[int] | bitarray``
        Input sequence. If a ``str``, must contain only ``"0"`` or ``"1"``. If
        an iterable of integers, must contain only ``0`` or ``1``.
    tempsize : ``Optional[int]``
        Size of template. If ``None``, a suitable default will be found.
    blocksize : ``Optional[int]``
        Size of the blocks that partition the given sequence. If ``None``, a
        suitable default will be found.

    Returns
    -------
    ``dict[Any, tuple[int | float, float]]``
        The test's statistic and p-value mapped to each template.
    """
    _check_mutual_kwargs(tempsize, "tempsize", blocksize, "blocksize")
    a = frozenbitarray(bits)
    n = len(a)
    _check_bits_size(n, 2)
    if tempsize is None and blocksize is None:
        blocksize = max(ceil(0.0125 * n), 1)
        print(f"{blocksize=}")
        tempsize = min(max(blocksize // 3, 1), 10)
        print(f"{tempsize=}")
    nblocks = n // blocksize
    _check_recommendations(
        (n >= 100, f"{n=}", "n ≥ 100"),
        (tempsize in (9, 10), f"{tempsize=}", "tempsize == 9 or 10"),
        (blocksize > 0.01 * n, f"{n=} and {blocksize=}", "blocksize > 0.01 * n"),
    )

    block_counts = defaultdict(lambda: defaultdict(int))
    for i, chunk in enumerate(_chunked(a, nblocks, blocksize)):
        matches = defaultdict(int)
        for window in _windowed(chunk, tempsize):
            matches[window] += 1
        for temp, count in matches.items():
            block_counts[temp][i] = count

    count_expect = (blocksize - tempsize + 1) / 2 ** tempsize
    variance = blocksize * (
        (1 / 2 ** tempsize) - ((2 * tempsize - 1) / 2 ** (2 * tempsize))
    )
    results = ResultsMap()
    for temp in _product(tempsize):
        count_diffs = [block_counts[temp][b] - count_expect for b in range(nblocks)]
        chi2 = sum(diff ** 2 / variance for diff in count_diffs)
        p = gammaincc(nblocks / 2, chi2 / 2)
        results[temp] = Result(chi2, p)

    return results


def otm(
    bits, tempsize: Optional[int] = None, blocksize: Optional[int] = None
) -> Result:
    """Overlapping matches to template per block is compared to expected result

    Parameters
    ----------
    bits : ``str | Iterable[int] | bitarray``
        Input sequence. If a ``str``, must contain only ``"0"`` or ``"1"``. If
        an iterable of integers, must contain only ``0`` or ``1``.
    tempsize : ``Optional[int]``
        Size of template. If ``None``, a suitable default will be found.
    blocksize : ``Optional[int]``
        Size of the blocks that partition the given sequence. If ``None``, a
        suitable default will be found.

    Returns
    -------
    ``tuple[int | float, float]``
        The test's statistic and p-value.
    """
    _check_mutual_kwargs(tempsize, "tempsize", blocksize, "blocksize")
    a = frozenbitarray(bits)
    n = len(a)
    _check_bits_size(n, 2)
    if tempsize is None and blocksize is None:
        blocksize = max(floor(sqrt(n)), 1)
        tempsize = min(max(floor(sqrt(blocksize)), 1), 12)
    nblocks = n // blocksize
    _check_recommendations(
        (n >= 288, f"{n=}", "n ≥ 288"),
        (
            tempsize - log(nblocks, 2) < log(n),
            "{tempsize=} and {nblocks=}",
            "tempsize ≈ log2(nblocks)",
        ),
    )
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


def universal(
    bits, blocksize: Optional[int] = None, init_nblocks: Optional[int] = None
) -> Result:
    """Distance between patterns is compared to expected result

    Parameters
    ----------
    bits : ``str | Iterable[int] | bitarray``
        Input sequence. If a ``str``, must contain only ``"0"`` or ``"1"``. If
        an iterable of integers, must contain only ``0`` or ``1``.
    blocksize : ``Optional[int]``
        Size of the blocks that partition the given sequence. If ``None``, a
        suitable default will be found.
    init_nblocks : ``Optional[int]``
        Number of blocks to be used in identifying patterns, beginning at the
        start of the sequence. If ``None``, a suitable default will be found.

    Returns
    -------
    ``tuple[int | float, float]``
        The test's statistic and p-value.
    """
    _check_mutual_kwargs(blocksize, "blocksize", init_nblocks, "init_nblocks")
    a = frozenbitarray(bits)
    n = len(a)
    _check_bits_size(n, 4)
    if blocksize is None and init_nblocks is None:
        # TODO: remove magic numbers (values from SP800-22, section 2.9.7, p. 45)
        defaults = {
            # n: (blocksize, init_nblocks)
            387_840: (6, 640),
            904_960: (7, 1280),
            2_068_480: (8, 2560),
            4_654080: (9, 5120),
            10_342_400: (10, 10240),
            22_753_280: (11, 20480),
            49_643_520: (12, 40960),
            107_560_960: (13, 81920),
            231_669_760: (14, 163840),
            496_435_200: (15, 327680),
            105_9061_760: (16, 655360),
        }
        try:
            key = max(k for k in defaults.keys() if k <= n)
            blocksize, init_nblocks = defaults[key]
        except ValueError:
            blocksize = min(max(ceil(log(n)), 2), 16)
            nblocks = n // blocksize
            init_nblocks = max(nblocks // 100, 1)
    nblocks = n // blocksize
    test_nblocks = nblocks - init_nblocks
    _check_recommendations(
        (n >= 387_840, f"{n=}", "n ≥ 387840"),
        (6 <= blocksize <= 16, f"{blocksize=}", "6 ≤ blocksize ≤ 16"),
        (
            init_nblocks - 10 * 2 ** test_nblocks < log(n),
            f"{init_nblocks=} and {test_nblocks=}",
            "init_nblocks ≈ 10 * 2 ** test_nblocks",
        ),
    )
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


def complexity(bits: Bits, blocksize: Optional[int] = None) -> Result:
    """LSFRs of blocks is compared to expected length

    Parameters
    ----------
    bits : ``str | Iterable[int] | bitarray``
        Input sequence. If a ``str``, must contain only ``"0"`` or ``"1"``. If
        an iterable of integers, must contain only ``0`` or ``1``.
    blocksize : ``Optional[int]``
        Size of the blocks that partition the given sequence. If ``None``, a
        suitable default will be found.

    Returns
    -------
    ``tuple[int | float, float]``
        The test's statistic and p-value.
    """
    a = frozenbitarray(bits)
    n = len(a)
    _check_bits_size(n, 4)
    if blocksize is None:
        blocksize = max(floor(sqrt(n)), 2)
    nblocks = n // blocksize
    _check_recommendations(
        (n >= 1_000_000, f"{n=}", "n ≥ 1mil"),
        (500 <= blocksize <= 5000, f"{blocksize=}", "500 ≤ blocksize ≤ 5000"),
        (nblocks >= 200, f"{nblocks=}", "nblocks ≥ 200"),
    )
    a = a[: nblocks * blocksize]
    mean_expect = (
        blocksize / 2
        + (9 + (-(1 ** (blocksize + 1)))) / 36
        - (blocksize / 3 + 2 / 9) / 2 ** blocksize
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


def serial(bits: Bits, blocksize: Optional[int] = None) -> Tuple[Result, Result]:
    """Proportion of all overlapping patterns is compared to expected uniformity

    Parameters
    ----------
    bits : ``str | Iterable[int] | bitarray``
        Input sequence. If a ``str``, must contain only ``"0"`` or ``"1"``. If
        an iterable of integers, must contain only ``0`` or ``1``.
    blocksize : ``Optional[int]``
        Size of the blocks that partition the given sequence. If ``None``, a
        suitable default will be found.

    Returns
    -------
    ``tuple[int | float, float]``
        The test's statistic and p-value.
    """
    a = frozenbitarray(bits)
    n = len(a)
    _check_bits_size(n, 2)
    if blocksize is None:
        blocksize = max(floor(log(n, 2)) - 3, 2)
    _check_recommendations(
        (
            blocksize < floor(log(n, 2)) - 2,
            f"{n=} and {blocksize=}",
            "blocksize < ⌊log2(n) - 2⌋",
        )
    )

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
            norm_sums[tempsize] = 0

    norm_sum_delta1 = norm_sums[blocksize] - norm_sums[blocksize - 1]
    p1 = gammaincc(2 ** (blocksize - 2), norm_sum_delta1 / 2)
    normsum_delta2 = (
        norm_sums[blocksize] - 2 * norm_sums[blocksize - 1] + norm_sums[blocksize - 2]
    )
    p2 = gammaincc(2 ** (blocksize - 3), normsum_delta2 / 2)

    return ResultsTuple((Result(norm_sum_delta1, p1), Result(normsum_delta2, p2)))


def apen(bits: Bits, blocksize: Optional[int] = None) -> Result:
    """Approximate entropy of sequence is compared to expected result

    Parameters
    ----------
    bits : ``str | Iterable[int] | bitarray``
        Input sequence. If a ``str``, must contain only ``"0"`` or ``"1"``. If
        an iterable of integers, must contain only ``0`` or ``1``.
    blocksize : ``Optional[int]``
        Size of the blocks that partition the given sequence. If ``None``, a
        suitable default will be found.

    Returns
    -------
    ``tuple[int | float, float]``
        The test's statistic and p-value.
    """
    a = frozenbitarray(bits)
    n = len(a)
    _check_bits_size(n, 2)
    if blocksize is None:
        blocksize = max(floor(log(n, 2)) - 6, 2)
    _check_recommendations(
        (
            blocksize < floor(log(n, 2)) - 5,
            f"{n=} and {blocksize=}",
            "blocksize < ⌊log2(n)⌋ - 5",
        )
    )

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


def cumsum(bits: Bits, reverse: bool = False) -> Result:
    """Furthest detour in a randomn walk is compared to expected result

    Parameters
    ----------
    bits : ``str | Iterable[int] | bitarray``
        Input sequence. If a ``str``, must contain only ``"0"`` or ``"1"``. If
        an iterable of integers, must contain only ``0`` or ``1``.
    reverse : ``bool``
        If ``True``, cumulate from the end of the sequence first.

    Returns
    -------
    ``tuple[int | float, float]``
        The test's statistic and p-value.
    """
    a = frozenbitarray(bits)
    n = len(a)
    _check_bits_size(n, 2)
    _check_recommendations((n >= 100, f"{n=}", "n ≥ 100"))

    o = _oscillate(a)
    if reverse:
        o = o[::-1]
    sums = np.cumsum(o)
    abs_sums = np.abs(sums)

    max_sum = abs_sums.max()
    # TODO: do this more declaratively to remove the need for eye bleach
    p = (
        1
        - sum(
            norm.cdf((4 * k + 1) * max_sum / sqrt(n))
            - norm.cdf((4 * k - 1) * max_sum / sqrt(n))
            for k in np.arange(
                floor((-n / max_sum + 1) / 4), floor((n / max_sum - 1) / 4) + 1, 1
            )
        )
        + sum(
            norm.cdf((4 * k + 3) * max_sum / sqrt(n))
            - norm.cdf((4 * k + 1) * max_sum / sqrt(n))
            for k in np.arange(
                floor((-n / max_sum - 3) / 4), floor((n / max_sum - 1) / 4) + 1, 1
            )
        )
    )

    return Result(max_sum, p)


def excursions(bits: Bits) -> Dict[int, Result]:
    """Frequency of states per cycle in a random walk is compared to expected
    results

    Parameters
    ----------
    bits : ``str | Iterable[int] | bitarray``
        Input sequence. If a ``str``, must contain only ``"0"`` or ``"1"``. If
        an iterable of integers, must contain only ``0`` or ``1``.

    Returns
    -------
    ``dict[Any, tuple[int | float, float]]``
        The test's statistic and p-value mapped to each w/e.
    """
    a = frozenbitarray(bits)
    n = len(a)
    _check_bits_size(n, 2)
    _check_recommendations((n >= 1_000_000, f"{n=}", "n ≥ 1mil"))

    o = _oscillate(a)
    sums = np.cumsum(o)
    states = (-4, -3, -2, -1, 1, 2, 3, 4)
    intervals = (0, 1, 2, 3, 4, 5)
    state_count_bins = {state: {i: 0 for i in intervals} for state in states}
    ncycles = 0
    for cycle in _ascycles(sums):
        ncycles += 1
        state_counts = defaultdict(
            int, dict(zip(*np.unique(cycle, return_counts=True)))
        )
        for state in states:
            count = state_counts[state]
            state_count_bins[state][_binkey(intervals, count)] += 1

    state_count_probs = {
        # state: <bin interval probabilities>
        1: [0.5000, 0.2500, 0.1250, 0.0625, 0.0312, 0.0312],
        2: [0.7500, 0.0625, 0.0469, 0.0352, 0.0264, 0.0791],
        3: [0.8333, 0.0278, 0.0231, 0.0193, 0.0161, 0.0804],
        4: [0.8750, 0.0156, 0.0137, 0.0120, 0.0105, 0.0733],
        # 5: [0.9000, 0.0100, 0.0090, 0.0081, 0.0073, 0.0656],
        # 6: [0.9167, 0.0069, 0.0064, 0.0058, 0.0053, 0.0588],
        # 7: [0.9286, 0.0051, 0.0047, 0.0044, 0.0041, 0.0531],
    }

    results = ResultsMap()
    for state in states:
        expected_bin_counts = [ncycles * prob for prob in state_count_probs[abs(state)]]
        chi2, p = chisquare(list(state_count_bins[state].values()), expected_bin_counts)
        results[state] = Result(chi2, p)

    return results


def vexcursions(bits: Bits) -> Dict[int, Result]:
    """Proportion of states in a random walk is compared to expected results

    Parameters
    ----------
    bits : ``str | Iterable[int] | bitarray``
        Input sequence. If a ``str``, must contain only ``"0"`` or ``"1"``. If
        an iterable of integers, must contain only ``0`` or ``1``.

    Returns
    -------
    ``dict[Any, tuple[int | float, float]]``
        The test's statistic and p-value mapped to each w/e.
    """
    a = frozenbitarray(bits)
    n = len(a)
    _check_bits_size(n, 2)
    _check_recommendations((n >= 1_000_000, f"{n=}", "n ≥ 1mil"))

    o = _oscillate(a)
    sums = np.cumsum(o)
    state_counts = defaultdict(int, dict(zip(*np.unique(sums, return_counts=True))))
    ncycles = state_counts[0] + 1

    results = ResultsMap()
    for state in (-9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9):
        count = state_counts[state]
        p = erfc(abs(count - ncycles) / sqrt(2 * ncycles * (4 * abs(state) - 2)))
        results[state] = Result(count, p)

    return results
