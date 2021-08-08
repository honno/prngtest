from bitarray.util import urandom
from pytest import mark, param

from prngtest import *

# TODO make this reproducible
a = urandom(1_000_000)


# TODO remove this and call functions with no kwargs once they can be inferred
def e(randtest, **kwargs):
    if len(kwargs) == 0:
        name = randtest.__name__
    else:
        f_kwargs = ", ".join(f"{k}={repr(v)}" for k, v in kwargs.items())
        name = f"{randtest.__name__}({f_kwargs})"
    return param(randtest, kwargs, id=name)


@mark.parametrize(
    "randtest, kwargs",
    [
        e(monobit),
        e(block_frequency),
        e(runs),
        e(longest_runs),
        e(matrix_rank),
        e(spectral),
        e(otm),
        e(universal),
        e(complexity),
        e(serial),
        e(apen),
        e(cusum),
        e(excursions),
        e(excursions_variant),
    ]
)
def test_randtests_pass_random_bits(randtest, kwargs):
    result = randtest(a, **kwargs)
    assert result.p > 0.01


@mark.parametrize(
    "randtest, kwargs",
    [
        e(notm, tempsize=9, blocksize=10000),
    ]
)
def test_multi_randtests_all_pass_random_bits(randtest, kwargs):
    result = randtest(a, **kwargs)
    for p in result.pvalues:
        assert p > 0.01
