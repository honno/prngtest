import pytest
from bitarray.util import urandom

from prngtest import *

# TODO use SHA-1 data
a = urandom(1_000_000)
a[1] = 1 - a[0]  # ensures mixed bits when slicing


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "randtest, min_n",
    [
        (monobit, 2),
        (blockfreq, 8),
        (runs, 2),
        (blockruns, 128),
        (matrix, 4),
        (spectral, 2),
        (notm, 2),
        (otm, 2),
        (universal, 4),
        (complexity, 4),
        (serial, 2),
        (apen, 2),
        (cumsum, 2),
        (excursions, 2),
        (vexcursions, 2),
    ],
)
def test_randtests_can_default_kwargs(randtest, min_n):
    randtest(a[:min_n])


@pytest.mark.parametrize(
    "randtest",
    [
        monobit,
        blockfreq,
        runs,
        blockruns,
        matrix,
        spectral,
        otm,
        universal,
        pytest.param(complexity, marks=pytest.mark.slow),
        apen,
        cumsum,
    ],
)
def test_randtests_pass_random_bits(randtest):
    result = randtest(a)
    assert result.p > 0.01


@pytest.mark.parametrize("randtest", [notm, serial, excursions, vexcursions])
def test_mapped_randtests_pass_random_bits(randtest):
    results = randtest(a)
    for p in results.pvalues:
        assert p > 0.01
