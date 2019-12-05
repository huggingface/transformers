# content of conftest.py

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--use_cuda", action="store_true", default=False, help="run tests on gpu"
    )


@pytest.fixture
def use_cuda(request):
    """ Run test on gpu """
    return request.config.getoption("--use_cuda")
