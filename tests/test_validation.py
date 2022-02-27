import pytest

from prngtest import *

from . import constants


@pytest.mark.parametrize(
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
        vexcursions,
    ],
)
def test_error_on_single_bit(randtest):
    with pytest.raises(ValueError):
        randtest("0")


@pytest.mark.parametrize(
    "randtest, n",
    [
        (monobit, 99),
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
        (vexcursions, 10),
    ],
)
def test_warn_on_disapproved_input(randtest, n):
    with pytest.warns(UserWarning):
        randtest(constants.e[:n])
