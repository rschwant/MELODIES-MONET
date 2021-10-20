"""
pytest test_model.py
"""
import sys
import numpy as np
sys.path.insert(0, '..')
import driver

"""
define a variable in the global scope
to hold a driver.model object
and be shared across unit tests
"""
model = None
file_str = '*.nc'
file_test = 'test_model.nc'


def test_init():
    global model
    model = driver.model()
    model.file_str = file_str
    assert True


def test_glob_files():
    global model
    model.glob_files()
    print(model.files)
    assert type(model.files) is np.ndarray
    assert model.files[0] == file_test
