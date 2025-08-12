from delphi.data.core import build_datasets

train_ds, val_ds = build_datasets(data_dict={"cohort": "mimic", "block_size": 1024})

print()
