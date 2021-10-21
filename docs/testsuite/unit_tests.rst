Unit Tests
==========

The unit tests are intended to test specific functionality
within a MELODIES python class.
MELODIES-MONET uses the `pytest <http://www.pytest.org>`__ framework
which can be installed into the python environment via::

    $ conda install pytest

The test files are located in melodies_monet/tests.
In general there is one file per class,
and within each file there is defined a test for each class method.

The unit tests use generic random test data to initialize a class.
This data needs to be created before running pytest.
To generate random test data for a given YAML control file::

    $ python test_setup.py --control control.yaml

To test a single class, e.g. driver.model::

    $ pytest test_model.py --control control.yaml
