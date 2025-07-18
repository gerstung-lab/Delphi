from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from utils import assessment_age, build_biomarker, build_dataframe

panel_dir = Path("data/gather_biomarker/panel")

prs_panel = [
    26206,  # alzheimer
    26204,  # AMD
    26210,  # asthma
    26212,  # atrial_fibrillation
    26214,  # bipolar_disorder
    26216,  # body_mass_index
    26234,  # bone_mineral_density
    26218,  # bowel_cancer
    26220,  # breast_cancer
    26227,  # cad
    26223,  # cardiovascular_disease
    26225,  # coeliac_disease
    26229,  # crohns
    26232,  # epithelial_ovarian_cancer
    26265,  # glaucoma
    26238,  # glycated_haemoglobin
    26242,  # HDL
    26240,  # height
    26244,  # hypertension
    26246,  # intraocular_pressure
    26248,  # ischaemic_stroke
    26250,  # LDL
    26278,  # lupus
    26252,  # melanoma
    26202,  # menopause
    26254,  # multiple_sclerosis
    26258,  # osteoporosis
    26260,  # parkinsons
    26267,  # prostate_cancer
    26269,  # psoriasis
    26273,  # rheumatoid_arthritis
    26275,  # schizophrenia
    26283,  # T1D
    26285,  # T2D
    26287,  # ulcerative_colitis
    26289,  # venous_thromboembolic_disease
]
prs_df = build_dataframe(fids=prs_panel, visit_idx=0)
zero_age = pd.Series(data=0, index=prs_df.index)
build_biomarker(biomarker_df=prs_df, time_df=zero_age, biomarker="prs")


blood_panel_dir = panel_dir / "blood"
blood_panel_yamls = blood_panel_dir.glob("*.yaml")
for blood_panel_yaml in blood_panel_yamls:
    blood_panel = blood_panel_yaml.stem
    with open(blood_panel_yaml, "r") as file:
        panel_biomarkers = yaml.safe_load(file)
    print(blood_panel)
    df = build_dataframe(fids=panel_biomarkers, visit_idx=0)
    build_biomarker(biomarker_df=df, time_df=assessment_age, biomarker=blood_panel)


urine_panels = {
    "nak": [30530, 30520],  # sodium  # potassium
    "creat": [30510],  # creatinine
    "albu": [30500],  # microalbumin
}
for urine_panel_name, urine_panel in urine_panels.items():
    df = build_dataframe(fids=urine_panel, visit_idx=0)
    build_biomarker(biomarker_df=df, time_df=assessment_age, biomarker=urine_panel_name)


telomere_panel = [22191]  # adjusted_ratio
telomere_df = build_dataframe(fids=telomere_panel, visit_idx=0)
build_biomarker(biomarker_df=telomere_df, time_df=assessment_age, biomarker="telomere")


food_panel = [
    1369,  # beef
    1438,  # bread
    1458,  # cereal
    1498,  # coffee
    1289,  # cooked_vegetable
    1319,  # dried_fruit
    1309,  # fresh_fruit
    1379,  # lamb
    1339,  # non_oily_fish
    1329,  # oily_fish
    1389,  # pork
    1359,  # poultry
    1349,  # processed_meat
    1299,  # salad
    1488,  # tea
    1528,  # water
]
food_df = build_dataframe(fids=food_panel, visit_idx=0)
food_df = food_df.replace(
    {
        -3: np.nan,  # prefer not answer
        -1: np.nan,  # do not know
        -10: 0,  # less than one
    }
)
build_biomarker(biomarker_df=food_df, time_df=assessment_age, biomarker="diet")


met_panel = [22037, 22038, 22039]  # walking  # moderate  # vigorous
met_df = build_dataframe(fids=met_panel, visit_idx=0)
met_df /= 24 * 60
build_biomarker(biomarker_df=met_df, time_df=assessment_age, biomarker="met")
