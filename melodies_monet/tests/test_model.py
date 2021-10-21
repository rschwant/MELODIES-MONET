"""
pytest test_model.py
"""
import sys
import numpy as np
import xarray as xr
sys.path.insert(0, '..')
import driver

"""
define a variable in the global scope
to hold a driver.model object
and YAML control object
and be shared across unit tests
"""
model = None
control = None

file_str = '*.nc'
file_test = 'test_model.nc'



def test_read_control_yaml(control_yaml):
    import yaml
    global control
    with open(control_yaml, 'r') as f:
        control = yaml.safe_load(f)


def test_init():
    global model
    model = driver.model()
    model.file_str = file_str
    model.obj = xr.open_dataset(file_test)
    model.variable_dict = control['model']['test_model']['variables']
    print(model.variable_dict)
    assert True


def test_glob_files():
    global model
    model.glob_files()
    print(model.files)
    assert type(model.files) is np.ndarray
    assert model.files[0] == file_test


def test_mask_and_scale():
    global model
    model.mask_and_scale()
    assert True
