from math import isclose

import pytest
from bitarray import frozenbitarray

from prngtest import *

from . import constants

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def statistic_isclose(statistic, statistic_expect):
    if isinstance(statistic, int):
        return statistic == statistic_expect
    else:
        return isclose(statistic, statistic_expect, rel_tol=0.05)


def p_isclose(p, p_expect):
    return isclose(p, p_expect, abs_tol=0.005)


def e(randtest, bits, statistic, p, *, xfail=False, slow=False, **kwargs):
    if len(kwargs) == 0:
        name = f"{randtest.__name__}(<{len(bits)} bits>)"
    else:
        f_kwargs = ", ".join(f"{k}={repr(v)}" for k, v in kwargs.items())
        name = f"{randtest.__name__}(<{len(bits)} bits>, {f_kwargs})"

    marks = []
    if xfail:
        marks.append(pytest.mark.xfail)
    if slow:
        marks.append(pytest.mark.slow)

    return pytest.param(randtest, bits, kwargs, statistic, p, id=name, marks=marks)


@pytest.mark.parametrize(
    "randtest, bits, kwargs, statistic, p",
    [
        e(
            randtest=monobit,
            bits="1011010101",
            statistic=0.632455532,
            p=0.527089,
        ),
        e(
            randtest=monobit,
            bits=(
                "11001001"
                "00001111"
                "11011010"
                "10100010"
                "00100001"
                "01101000"
                "11000010"
                "00110100"
                "11000100"
                "11000110"
                "01100010"
                "10001011"
                "1000"
            ),
            statistic=1.6,
            p=0.109599,
        ),
        e(
            randtest=blockfreq,
            bits=(
                "1100100100"
                "0011111101"
                "1010101000"
                "1000100001"
                "0110100011"
                "0000100011"
                "0100110001"
                "0011000110"
                "0110001010"
                "0010111000"
            ),
            blocksize=10,
            statistic=7.2,
            p=0.706438,
        ),
        e(
            randtest=runs,
            bits=("10011010" "11"),
            statistic=7,
            p=0.147232,
        ),
        e(
            randtest=blockruns,
            bits=(
                "11001100"
                "00010101"
                "01101100"
                "01001100"
                "11100000"
                "00000010"
                "01001101"
                "01010001"
                "00010011"
                "11010110"
                "10000000"
                "11010111"
                "11001100"
                "11100110"
                "11011000"
                "10110010"
            ),
            statistic=4.882605,
            p=0.180609,
        ),
        e(
            randtest=matrix,
            bits=("01011001" "00101010" "1101"),
            nrows=3,
            ncols=3,
            statistic=0.596953,
            p=0.741948,
        ),
        e(
            randtest=matrix,
            bits=constants.e[:100_000],
            nrows=32,
            ncols=32,
            statistic=1.2619656,
            p=0.532069,
        ),
        e(
            randtest=spectral,
            bits="1001010011",
            statistic=-2.176429,
            p=0.029523,
            # DFT in sts has 1 peak above threshold, should be 0
            # also we use FFT and not a naive approach
            xfail=True,
        ),
        e(
            randtest=spectral,
            bits=(
                "11001001"
                "00001111"
                "11011010"
                "10100010"
                "00100001"
                "01101000"
                "11000010"
                "00110100"
                "11000100"
                "11000110"
                "01100010"
                "10001011"
                "1000"
            ),
            statistic=-1.376494,
            p=0.168669,
            xfail=True,
        ),
        e(
            randtest=otm,
            bits=(
                "1011101111"
                "0010110110"  # Modifed 2nd block of SP800-22 e
                "0111001011"  # originally had 1 match
                "1011111000"  # now has 2 matches, as expected
                "0101101001"
            ),
            tempsize=2,
            blocksize=10,  # nblocks=5
            df=2,
            statistic=3.167729,
            p=0.274932,
            # p off by ~0.07 if gammaincc(df/2, statistic/2) w/ df=2
            xfail=True,
        ),
        e(
            randtest=otm,
            bits=constants.e,
            tempsize=9,
            blocksize=1032,  # nblocks=968
            statistic=8.965859,
            p=0.110434,
        ),
        e(
            randtest=universal,
            bits=("01011010" "01110101" "0111"),
            blocksize=2,
            init_nblocks=4,
            statistic=1.1949875,
            p=0.767189,
        ),
        e(
            randtest=complexity,
            bits=constants.e,
            blocksize=1000,
            statistic=2.700348,
            p=0.845406,
            slow=True,
        ),
        e(
            randtest=apen,
            bits="0100110101",
            blocksize=3,
            statistic=10.043859999999999,  # SP800-22 erroneously had 0.502193
            p=0.261961,
        ),
        e(
            randtest=apen,
            bits=(
                "11001001"
                "00001111"
                "11011010"
                "10100010"
                "00100001"
                "01101000"
                "11000010"
                "00110100"
                "11000100"
                "11000110"
                "01100010"
                "10001011"
                "1000"
            ),
            blocksize=2,
            statistic=5.550792,
            p=0.235301,
        ),
        e(
            randtest=cumsum,
            bits="1011010111",
            statistic=4,
            p=0.4116588,
        ),
        e(
            randtest=cumsum,
            bits=(
                "11001001"
                "00001111"
                "11011010"
                "10100010"
                "00100001"
                "01101000"
                "11000010"
                "00110100"
                "11000100"
                "11000110"
                "01100010"
                "10001011"
                "1000"
            ),
            statistic=16,
            p=0.219194,
        ),
        e(
            randtest=cumsum,
            bits=(
                "11001001"
                "00001111"
                "11011010"
                "10100010"
                "00100001"
                "01101000"
                "11000010"
                "00110100"
                "11000100"
                "11000110"
                "01100010"
                "10001011"
                "1000"
            ),
            reverse=True,
            statistic=19,
            p=0.114866,
        ),
    ],
)
def test_examples(randtest, bits, kwargs, statistic, p):
    result = randtest(bits, **kwargs)
    assert statistic_isclose(result.statistic, statistic)
    assert p_isclose(result.p, p)


@pytest.mark.parametrize(
    "randtest, bits, kwargs, statistics, pvalues",
    [
        e(
            randtest=serial,
            bits="0011011101",
            blocksize=3,
            statistic=[1.6, 0.8],
            p=[0.9057, 0.8805],
            # SP800-22's result is not replicated by sts
            xfail=True,
        ),
        e(
            randtest=serial,
            bits=constants.e,
            blocksize=2,
            statistic=[0.339764, 0.336400],
            p=[0.843764, 0.561915],
        ),
        e(
            randtest=excursions,
            bits=constants.e,
            statistic=[
                3.835698,
                7.318707,
                7.861927,
                15.692617,
                2.485906,
                5.429381,
                2.404171,
                2.393928,
            ],
            p=[
                0.573306,
                0.197996,
                0.164011,
                0.007779,
                0.778616,
                0.365752,
                0.790853,
                0.792378,
            ],
            # SP800-22's result is not replicated by sts
            xfail=True,
        ),
        e(
            randtest=vexcursions,
            bits=constants.e,
            statistic=[
                1450,
                1435,
                1380,
                1366,
                1412,
                1475,
                1480,
                1468,
                1502,
                1409,
                1369,
                1396,
                1479,
                1599,
                1628,
                1619,
                1620,
                1610,
            ],
            p=[
                0.858946,
                0.794755,
                0.576249,
                0.493417,
                0.633873,
                0.917283,
                0.934708,
                0.816012,
                0.826009,
                0.137861,
                0.200642,
                0.441254,
                0.939291,
                0.505683,
                0.445935,
                0.512207,
                0.538635,
                0.593930,
            ],
        ),
    ],
)
def test_mapped_examples(randtest, bits, kwargs, statistics, pvalues):
    result = randtest(bits, **kwargs)
    for statistic, statistic_expect in zip(result.statistics, statistics):
        assert statistic_isclose(statistic, statistic_expect)
    for p, p_expect in zip(result.pvalues, pvalues):
        assert p_isclose(p, p_expect)


def test_notm_sub_example():
    results = notm("10100100101110010110", tempsize=3, blocksize=10)
    result = results[frozenbitarray("001")]
    assert statistic_isclose(result.statistic, 2.133333)
    assert p_isclose(result.p, 0.344154)


def test_excursions_sub_example():
    results = excursions("0110110101")
    result = results[1]
    assert statistic_isclose(result.statistic, 4.333033)
    assert p_isclose(result.p, 0.502529)


def test_vexcursions_sub_example():
    results = vexcursions("0110110101")
    result = results[1]
    assert statistic_isclose(result.statistic, 4)
    assert p_isclose(result.p, 0.683091)
