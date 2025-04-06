import os

import numpy as np
import pandas as pd
import torch

from delphi.data import Dataset, data_loader, get_p2i
from delphi.distributed import para_dtype
from delphi.model.transformer import Delphi, DelphiConfig
from delphi.transform import AddNoEvent
from delphi.utils import get_batch

out_dir = "checkpoints/Delphi-demo"
device = "cpu"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = "float32"  #'bfloat16' # 'float32' or 'bfloat16' or 'float16'
seed = 1337

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device_type = "cuda" if "cuda" in device else "cpu"
dtype = para_dtype[dtype]

ckpt_path = os.path.join(out_dir, "ckpt.pt")
checkpoint = torch.load(ckpt_path, map_location=device)
conf = DelphiConfig(**checkpoint["model_args"])
model = Delphi(conf)
state_dict = checkpoint["model"]
model.load_state_dict(state_dict)

model.eval()
model = model.to(device)

val = np.fromfile("data/ukb_simulated_data/val.bin", dtype=np.uint32).reshape(-1, 3)
val_p2i = get_p2i(val)
batch_idx = np.arange(128)

d = get_batch(
    batch_idx,
    val,
    val_p2i,
    select="left",
    block_size=1000,
    device="cpu",
    no_event_token_rate=0,
    cut_batch=True,
)

p = []
model.to(device)
batch_size = 512
with torch.no_grad():
    for d_batch in zip(*map(lambda x: torch.split(x.to(device), batch_size), d)):
        p.append(model(*d_batch)[0].cpu().detach())
p = torch.vstack(p)


val_dataset = Dataset(
    participants=pd.unique(val[:, 0]),
    tokens=val[:, 2] + 1,
    time_steps=val[:, 1],
    start_pos=val_p2i[:, 0],
    seq_len=val_p2i[:, 1],
)

_, X, T = val_dataset.get_batch(batch_idx)

X = torch.tensor(X, dtype=torch.long)
T = torch.tensor(T, dtype=torch.float)

mask_time = -10000.0
T[X == 0] = mask_time
s = torch.argsort(T, 1)
X = torch.gather(X, 1, s)
T = torch.gather(T, 1, s)

X_t0 = X[:, :-1]
T_t0 = T[:, :-1]
X_t1 = X[:, 1:]
T_t1 = T[:, 1:]
d_batch = (X_t0, T_t0, X_t1, T_t1)

p_test = model(*d_batch)[0].cpu().detach()

print("test")
# transform = AddNoEvent(val_dataset.tokenizer, no_event_interval=5, mode="random", max_age=100)
