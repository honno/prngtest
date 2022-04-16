import pytest

from prngtest import _cli

from . import constants


@pytest.mark.parametrize(
    "extra_args", [[], pytest.param(["--all"], marks=pytest.mark.slow)]
)
def test_smoke(tmp_path, extra_args):
    data_path = tmp_path / "data.bin"
    constants.sha1.tofile(open(data_path, "wb"))
    _cli(str(data_path), *extra_args)
