#!/usr/bin/env python
"""Calculate epochs from max_steps."""
import numpy as np
from pathlib import Path

# Load training data to count patients
data_path = Path('../data/ukb_simulated_data/train.bin')
if not data_path.exists():
    data_path = Path('data/ukb_simulated_data/train.bin')
    
data = np.memmap(data_path, dtype=np.uint32, mode='r').reshape(-1, 3)
n_records = len(data)

# Count unique patients
patient_ids = data[:, 0]
unique_patients = len(np.unique(patient_ids))

# With batch_size=128
batch_size = 128
steps_per_epoch = unique_patients // batch_size

print('=== Training Data Statistics ===')
print(f'Total records: {n_records:,}')
print(f'Unique patients: {unique_patients:,}')
print(f'Batch size: {batch_size}')
print(f'Steps per epoch: {steps_per_epoch:,}')
print()
print('=== Coverage with Different max_steps ===')
for max_steps in [5000, 10000, 20000, 50000]:
    epochs = max_steps / steps_per_epoch
    print(f'max_steps={max_steps:,}: {epochs:.2f} epochs')
