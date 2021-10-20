"""
pytest test_pair.py
"""
import sys
sys.path.insert(0, '..')
import driver

"""
define a variable in the global scope
to hold a driver.pair object
and be shared across unit tests
"""
pair = None


def test_init():
    global pair
    pair = driver.pair()
    assert True

