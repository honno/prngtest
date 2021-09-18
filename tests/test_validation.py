from pytest import mark, raises

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
        notm,
        serial,
        excursions,
        excursions_variant,
    ],
)
def test_error_on_single_bit(randtest):
    with raises(ValueError):
        randtest("0")
