import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from utils import MULTIMODAL_INPUT_DIR, MULTIMODAL_OUTPUT_DIR, all_ukb_participants

from delphi.data.lmdb import estimate_write_size, write_lmdb

urine_panel_dir = "data/gather_biomarker/urine/panel"
