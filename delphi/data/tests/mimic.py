from delphi.data.mimic import MIMICDataset

train_ds = MIMICDataset(input_dir="data/mimic/train", n_positions=1024)

print(train_ds[10300][0])
