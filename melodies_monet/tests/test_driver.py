"""
pytest test_driver.py
"""
import driver

"""
define a variable in the global scope
to hold a driver.analysis object
and be shared across unit tests
"""
an = None


def test_init():
    global an
    an = driver.analysis()
    assert True


def test_read_control():
    global an
    an.read_control()
    assert True
