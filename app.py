from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple, Union
import torch
import yaml
from model import DelphiConfig, Delphi
import pandas as pd
import numpy as np

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set up device and dtype
device = config['device']
dtype = {'float32': torch.float32, 'float64': torch.float64, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config['dtype']]
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

# Load model
ckpt_path = config['ckpt_path']
checkpoint = torch.load(ckpt_path, map_location=device)
conf = DelphiConfig(**checkpoint['model_args'])
model = Delphi(conf)
state_dict = checkpoint['model']
model.load_state_dict(state_dict)
model.eval()
model = model.to(device)

# Load labels
delphi_labels = pd.read_csv('delphi_labels_chapters_colours_icd.csv')
name_to_token_id = {row[1]['name']: row[1]['index'] for row in delphi_labels.iterrows()}

app = FastAPI(title="Delphi Health Trajectory Extrapolator")

class HealthEvent(BaseModel):
    event: str
    age: float

@app.get("/model_stats")
async def get_model_stats():
    return checkpoint['model_args']

@app.post("/extrapolate_trajectory")
async def extrapolate_trajectory(trajectory: List[HealthEvent], max_new_tokens: int = 100):
    try:
        # Convert to list of tuples
        traj_list = [(event.event, event.age) for event in trajectory]
        
        # Convert ages to days
        traj_days = [(a, b * 365.25) for a, b in traj_list]
        
        # Get events and ages
        events = [name_to_token_id.get(event[0], 0) for event in traj_days]  # Default to 0 if not found
        events = torch.tensor(events, device=device).unsqueeze(0)
        ages = [event[1] for event in traj_days]
        ages = torch.tensor(ages, device=device).unsqueeze(0)
        
        # Generate
        with torch.no_grad():
            y, b, _ = model.generate(events, ages, max_new_tokens, termination_tokens=[1269])
            # Convert to readable format
            events_data = zip(y.cpu().numpy().flatten(), b.cpu().numpy().flatten() / 365.)
            
            result = []
            for i, (event_id, age_years) in enumerate(events_data):
                event_name = delphi_labels.loc[event_id, 'name'] if event_id in delphi_labels.index else f"Unknown ({event_id})"
                result.append({"age": float(age_years), "event": event_name})
        
        return {"trajectory": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
