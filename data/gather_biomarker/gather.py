from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from utils import build_biomarker, build_longitudinal_df, longitudinal_assessment_age

panel_dir = Path("data/gather_biomarker/panel")

with open(panel_dir / "prs.yaml", "r") as f:
    prs_panel = yaml.safe_load(f)
prs_df = build_longitudinal_df(fids=prs_panel)
zero_age = pd.Series(data=0, index=prs_df.index)
build_biomarker(biomarker_df=prs_df, time_series=zero_age, biomarker="prs")


blood_panel_dir = panel_dir / "blood"
blood_panel_yamls = blood_panel_dir.glob("*.yaml")
assess_age = longitudinal_assessment_age(visits=["init_assess", "1st_repeat_assess"])
for blood_panel_yaml in blood_panel_yamls:
    blood_panel = blood_panel_yaml.stem
    with open(blood_panel_yaml, "r") as file:
        panel_biomarkers = yaml.safe_load(file)
    df = build_longitudinal_df(fids=[str(i) for i in panel_biomarkers])
    build_biomarker(biomarker_df=df, time_series=assess_age, biomarker=blood_panel)


urine_panels = {
    "nak": [30530, 30520],  # sodium  # potassium
    "creat": [30510],  # creatinine
    "albu": [30500],  # microalbumin
}
assess_age = longitudinal_assessment_age(visits=["init_assess", "1st_repeat_assess"])
for urine_panel_name, urine_panel in urine_panels.items():
    df = build_longitudinal_df(fids=urine_panel)
    build_biomarker(biomarker_df=df, time_series=assess_age, biomarker=urine_panel_name)


telomere_panel = [22191]  # adjusted_ratio
telomere_df = build_longitudinal_df(fids=telomere_panel)
assess_age = longitudinal_assessment_age(
    visits=["init_assess", "1st_repeat_assess", "img"]
)
build_biomarker(biomarker_df=telomere_df, time_series=assess_age, biomarker="telomere")


with open(panel_dir / "diet.yaml", "r") as f:
    food_panel = yaml.safe_load(f)
food_df = build_longitudinal_df(fids=food_panel)
food_df = food_df.replace(
    {
        -3: np.nan,  # prefer not answer
        -1: np.nan,  # do not know
        -10: 0,  # less than one
    }
)
assess_age = longitudinal_assessment_age(
    visits=["init_assess", "1st_repeat_assess", "img", "1st_repeat_img"]
)
build_biomarker(biomarker_df=food_df, time_series=assess_age, biomarker="diet")


met_panel = [22037, 22038, 22039]  # walking  # moderate  # vigorous
met_df = build_longitudinal_df(fids=met_panel)
met_df /= 24 * 60
assess_age = longitudinal_assessment_age(
    visits=["init_assess", "1st_repeat_assess", "img", "1st_repeat_img"]
)
build_biomarker(biomarker_df=met_df, time_series=assess_age, biomarker="met")
