from pytest import mark, raises, warns

from prngtest import (
    apen,
    blockfreq,
    blockruns,
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

from . import constants


@mark.parametrize(
    "randtest",
    [
        monobit,
        blockfreq,
        runs,
        blockruns,
        matrix,
        spectral,
        notm,
        otm,
        universal,
        complexity,
        serial,
        apen,
        cumsum,
        excursions,
        excursions_variant,
    ],
)
def test_error_on_single_bit(randtest):
    with raises(ValueError):
        randtest("0")


@mark.parametrize(
    "randtest, n",
    [
        (monobit, constants.e[:99]),
        (blockfreq, 99),
        (runs, 99),
        (matrix, 127),
        (spectral, 999),
        (notm, 99),
        (otm, 287),
        (complexity, 10),
        (serial, 10),
        (apen, 10),
        (cumsum, 99),
        (excursions, 10),
        (excursions_variant, 10),
    ],
)
def test_warn_on_disapproved_input(randtest, n):
    with warns(UserWarning):
        randtest(constants.e[:n])
