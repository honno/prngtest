import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--skip-slow",
        action="store_true",
        help="skip running slow test cases",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: test case takes relatively long to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip-slow"):
        for item in items:
            for mark in item.iter_markers():
                if mark.name == "slow":
                    item.add_marker(pytest.mark.skip(reason="disabled via --skip-slow"))
                    break
