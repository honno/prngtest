import pytest
from bitarray.util import urandom

from prngtest import *


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
    a = urandom(min_n)
    a[1] = 1 - a[0]  # ensure mixed bits
    randtest(a)
