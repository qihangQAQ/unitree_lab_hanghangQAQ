"""FDM task helpers and assets."""

import os
from importlib.util import find_spec

FDM_TASK_DIR = os.path.abspath(os.path.dirname(__file__))
"""Path to the local FDM task package."""

FDM_DATA_DIR = os.path.join(FDM_TASK_DIR, "data")
"""Path to copied FDM data assets."""

# Import robots to trigger task registration when Gym/IsaacLab is available.
if find_spec("gymnasium") is not None and find_spec("isaaclab") is not None:
    from .robots.g1 import *
