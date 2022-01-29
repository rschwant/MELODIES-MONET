import os
import sys
sys.path.append('../../')
sys.path.append('../../plots')
from satellite_swath_plots import make_swath
import driver

an = driver.analysis()
an.control = '../yaml/control_modis_l2.yaml'
an.read_control()
an.open_obs()

for obs_key in an.obs:
    label = an.obs[obs_key].label
    make_swath(an.obs[obs_key].obj)
