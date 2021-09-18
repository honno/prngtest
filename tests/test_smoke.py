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
    "randtest",
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
def test_randtests_pass_random_bits(randtest):
    result = randtest(a)
    assert result.p > 0.01


@mark.parametrize("randtest", [notm, serial, excursions, excursions_variant])
def test_mapped_randtests_pass_random_bits(randtest):
    results = randtest(a)
    for p in results.pvalues:
        assert p > 0.01
