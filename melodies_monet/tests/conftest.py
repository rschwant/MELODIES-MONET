import pytest

def pytest_addoption(parser):
    parser.addoption(
        '--control',
        default='control.yaml',
        action='store')

@pytest.fixture
def control_yaml(request):
    return request.config.getoption('--control')
