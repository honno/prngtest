from bitarray.util import urandom
from pytest import mark

import prngtest

# TODO make this reproducible
a = urandom(1_000_000)


@mark.parametrize(
    "randtest",
    [getattr(prngtest, name) for name in prngtest.__all__],
    ids=prngtest.__all__,
)
def test_randtests_pass_random_bits(randtest):
    result = randtest(a)
    assert result.p > 0.01
