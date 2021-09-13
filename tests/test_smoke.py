from bitarray.util import urandom
from pytest import mark

from prngtest import (
    apen,
    block_frequency,
    block_runs,
    complexity,
    cumsum,
    excursions,
    excursions_variant,
    matrix,
    monobit,
    notm,
    otm,
    runs,
    serial,
    spectral,
    universal,
)

# TODO make this reproducible
a = urandom(1_000_000)


@mark.parametrize(
    "randtest, kwargs",
    [
        monobit,
        block_frequency,
        runs,
        block_runs,
        matrix,
        spectral,
        otm,
        universal,
        complexity,
        apen,
        cumsum,
    ],
)
def test_randtests_pass_random_bits(randtest, kwargs):
    result = randtest(a, **kwargs)
    assert result.p > 0.01


@mark.parametrize(
    "randtest, kwargs",
    [
        notm,
        serial,
        excursions,
        excursions_variant,
    ],
)
def test_mapped_randtests_pass_random_bits(randtest, kwargs):
    result = randtest(a, **kwargs)
    for p in result.pvalues:
        assert p > 0.01
