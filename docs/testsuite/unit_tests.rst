Unit Tests
==========

The unit tests are intended to test specific functionality
within a MELODIES python class.
MELODIES-MONET uses the `pytest <http://www.pytest.org>`__ framework
which can be installed into the python environment via::

    $ conda install pytest

The test files are located in melodies_monet/tests.
To generate random test data for a given YAML control file::

    $ python test_setup.py --control control.yaml

To test a single class, e.g. driver.model::

    $ pytest test_model.py --control control.yaml
