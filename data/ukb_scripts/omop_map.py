import pandas as pd
import yaml
from utils import multimodal_dir

overlap_with_finngen = ["blood", "urine"]

ukb2omop = pd.read_csv(
    multimodal_dir / "all_concepts_numeric_prio.csv", index_col="field_id"
)

with open("data/gather_biomarker/panel.yaml", "r") as f:
    panel = yaml.safe_load(f)

omop_panel = {}

for biomarker_mod in overlap_with_finngen:
    for modality, fids in panel[biomarker_mod].items():
        omop_panel[modality] = [int(ukb2omop.loc[fid, "a8_concept"]) for fid in fids]  # type: ignore

with open("data/gather_biomarker/omop_panel.yaml", "w") as f:
    yaml.dump(dict(omop_panel), f, default_flow_style=False, sort_keys=False)
