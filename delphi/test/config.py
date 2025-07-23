import os
from pathlib import Path

import numpy as np

from delphi.env import DELPHI_DATA_DIR

DATASET = "ukb_real_data"
DATASET_DIR = Path(DELPHI_DATA_DIR) / DATASET

participant_list_dir = Path(DATASET_DIR) / "participants"
participants = set()
for participant_list in participant_list_dir.rglob("*.bin"):
    participants_np = np.fromfile(participant_list, dtype=np.uint32)
    participants = participants.union(set(participants_np.tolist()))

PARTICIPANTS = np.array(list(participants))

_, expansion_packs, _ = next(os.walk(DATASET_DIR / "expansion_packs"))
EXPANSION_PACK_PATHS = [
    os.path.join(DATASET_DIR, "expansion_packs", expansion_pack)
    for expansion_pack in expansion_packs
]
