import os
import pickle
import torch
from model import DelphiConfig, Delphi
from tqdm import tqdm
import pandas as pd
import numpy as np
import textwrap
import warnings
import shap

from utils import get_batch, get_p2i
from utils import shap_custom_tokenizer, shap_model_creator


delphi_labels = pd.read_csv('delphi_labels_chapters_colours_icd.csv')
labels = pd.read_csv("data/ukb_simulated_data/labels.csv", header=None, sep="\t")

out_dir = 'Delphi-2M'
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype ='float32' #'bfloat16' # 'float32' or 'bfloat16' or 'float16'
seed = 1337

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device_type = 'cuda' if 'cuda' in device else 'cpu'
dtype = {'float32': torch.float32, 'float64': torch.float64, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
conf = DelphiConfig(**checkpoint['model_args'])
model = Delphi(conf)
state_dict = checkpoint['model']
model.load_state_dict(state_dict)

model.eval()
model = model.to(device)


DATA_ROOT = # fill here

train = np.fromfile(f'{DATA_ROOT}/train.bin', dtype=np.uint32).reshape(-1,3)
val = np.fromfile(f'{DATA_ROOT}/val.bin', dtype=np.uint32).reshape(-1,3)

train_p2i = get_p2i(train)
val_p2i = get_p2i(val)


def get_person(idx):
    x, y, _, time = get_batch([idx], val, val_p2i,  
              select='left', block_size=64, 
              device=device, padding='random', 
              cut_batch=True)
    
    x, y = x[y > -1], y[y > -1]
    person = []
    for token_id, date in zip(x, y):
        person.append((id_to_token[token_id.item()], date.item()))
    return person, y, time[0][-1]


id_to_token = labels.to_dict()[0]
token_to_id = {v:k for k, v in id_to_token.items()}

def tokens_to_ids(tokens):
    return [token_to_id[t] for t in tokens]

def ids_to_tokens(ids):
    return [id_to_token[int(id_)] for id_ in ids]

def split_person(p):
    tokens = [i[0] for i in p]
    ages = [i[1] for i in p]
    return tokens, ages


shaply_val = []

for person_idx in tqdm(range(len(val_p2i))):
    try:
        person_to_process, time, time_target = get_person(person_idx)
        time_passed = (time_target - time).cpu().detach().numpy()
        
        person_tokens, person_ages = split_person(person_to_process)
        person_tokens_ids = tokens_to_ids(person_tokens)
        
        masker = shap.maskers.Text(shap_custom_tokenizer, output_type='str', mask_token='10000', collapse_mask_token=False)
        model_shap = shap_model_creator(model, labels.index.values, person_tokens_ids, person_ages, device)
        explainer = shap.Explainer(model_shap, masker, output_names=labels[0].values)
        
        shap_values = explainer([' '.join(list(map(lambda x: str(token_to_id[x]), person_tokens)))])
        shap_values.data = np.array([list(map(lambda x: f"{x[0]}({x[1]/365:.1f}) ", person_to_process))])
        shaply_val.append((person_tokens_ids, shap_values.values.astype(np.float16), time_passed, [person_idx] * len(person_tokens_ids)))
    except Exception as e:
        print(repr(e))
        pass


all_tokens = np.concatenate([i[0] for i in shaply_val])
all_values = np.concatenate([i[1] for i in shaply_val], axis=1)[0]
all_times_passed = np.concatenate([i[2] for i in shaply_val], axis=0)
all_people = np.concatenate([i[3] for i in shaply_val])


path = 'shap_agg.pickle'

with open(path, 'wb') as f:
    pickle.dump({'tokens': all_tokens, 'values': all_values, 'times': all_times_passed, 'model': out_dir, 'people': all_people}, f)

