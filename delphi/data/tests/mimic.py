from delphi.data.mimic.base import MIMICDataset

train_ds = MIMICDataset(input_dir="mimic", n_positions=1024)

print(train_ds[0])
