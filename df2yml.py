from dataclasses import asdict, dataclass

import pandas as pd
import yaml

from delphi.tokenizer import Disease, TokenizerSchema

tokenizer_df = pd.read_csv(
    "data/ukb_simulated_data/labels.csv", names=["disease"], sep="\t"
)

disease_mapping = {}
for i, (_, row) in enumerate(tokenizer_df.iterrows(), start=0):
    short_name = row["disease"].lower().replace(" ", "_")
    disease_mapping[short_name] = Disease(
        id=i,
        ukb_name=row["disease"],
    )
tokenizer = TokenizerSchema(version="0.1", mapping=disease_mapping)

with open("data/ukb_simulated_data/tokenizer.yaml", "w") as f:
    yaml.dump(asdict(tokenizer), f, default_flow_style=False, sort_keys=False)
