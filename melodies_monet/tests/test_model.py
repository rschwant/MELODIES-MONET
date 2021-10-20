"""
pytest test_model.py
"""
import sys
sys.path.insert(0, '..')
import driver

"""
define a variable in the global scope
to hold a driver.model object
and be shared across unit tests
"""
model = None


def test_init():
    global model
    model = driver.model()
    assert True

