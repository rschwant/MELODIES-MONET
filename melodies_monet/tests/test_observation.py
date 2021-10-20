"""
pytest test_observation.py
"""
import sys
sys.path.insert(0, '..')
import driver

"""
define a variable in the global scope
to hold a driver.observation object
and be shared across unit tests
"""
observation = None


def test_init():
    global observation
    observation = driver.observation()
    assert True

