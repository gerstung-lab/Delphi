import os

import numpy as np
import yaml

from delphi.tokenizer import load_tokenizer_from_yaml

tokenizer = load_tokenizer_from_yaml("data/ukb_simulated_data/tokenizer.yaml")

with open("config/disease_list/all.yaml", "r") as f:
    disease_lst = yaml.safe_load(f)

val_PTX = np.memmap("data/ukb_real_data/val.bin", dtype=np.uint32, mode="r").reshape(
    -1, 3
)

val_X = val_PTX[:, 2] + 1

disease_at_least = []
for disease in disease_lst:
    disease_token = tokenizer[disease]
    n_occur = np.sum(val_X == disease_token)
    if n_occur > 25:
        disease_at_least.append(disease)

print(f"Number of diseases with at least 25 occurrences: {len(disease_at_least)}")
with open("config/disease_list/all_at_least_25.yaml", "w") as f:
    yaml.dump(disease_at_least, f)
