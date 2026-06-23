"""FDM task helpers and assets."""

import os

FDM_TASK_DIR = os.path.abspath(os.path.dirname(__file__))
"""Path to the local FDM task package."""

FDM_DATA_DIR = os.path.join(FDM_TASK_DIR, "data")
"""Path to copied FDM data assets."""

# Import robots to trigger task registration
from .robots.g1 import *
