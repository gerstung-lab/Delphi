import numpy as np
import pandas as pd
from utils import (
    assessment_age,
    biomarkers,
    build_biomarker,
    index_by_visit,
    load_biomarker_df,
)

all_visits = ["init_assess", "1st_repeat_assess", "img", "1st_repeat_img"]
assess_age = assessment_age(visits=all_visits)
assess_age = index_by_visit(assess_age, visits=all_visits)

visits = ["img", "1st_repeat_img"]
abdo_comp_df = load_biomarker_df(fids=biomarkers["abdo_fat_long"], visits=visits)
build_biomarker(
    biomarker_df=abdo_comp_df, time_series=assess_age, biomarker="abdo_fat_long"
)


visits = ["img"]
abdo_comp_df = load_biomarker_df(fids=biomarkers["abdo_fat_cross"], visits=visits)
build_biomarker(
    biomarker_df=abdo_comp_df, time_series=assess_age, biomarker="abdo_fat_cross"
)


prs_df = load_biomarker_df(fids=biomarkers["prs"], visits=["birth"])
zero_age = pd.Series(data=0, index=prs_df.index)
build_biomarker(biomarker_df=prs_df, time_series=zero_age, biomarker="prs")


for blood_panel, markers in biomarkers["blood"].items():
    if blood_panel == "wbc":
        visits = ["init_assess", "1st_repeat_assess", "img"]
    else:
        visits = ["init_assess", "1st_repeat_assess"]
    df = load_biomarker_df(fids=markers, visits=visits)
    build_biomarker(biomarker_df=df, time_series=assess_age, biomarker=blood_panel)


for urine_panel, markers in biomarkers["urine"].items():
    df = load_biomarker_df(fids=markers, visits=["init_assess", "1st_repeat_assess"])
    build_biomarker(biomarker_df=df, time_series=assess_age, biomarker=urine_panel)


telomere_df = load_biomarker_df(
    fids=biomarkers["telomere"], visits=["init_assess", "1st_repeat_assess", "img"]
)
build_biomarker(biomarker_df=telomere_df, time_series=assess_age, biomarker="telomere")


diet_df = load_biomarker_df(
    fids=biomarkers["diet"],
    visits=["init_assess", "1st_repeat_assess", "img", "1st_repeat_img"],
)
diet_df = diet_df.replace(
    {
        -3: np.nan,  # prefer not answer
        -1: np.nan,  # do not know
        -10: 0,  # less than one
    }
)
build_biomarker(biomarker_df=diet_df, time_series=assess_age, biomarker="diet")


met_df = load_biomarker_df(
    fids=biomarkers["met"],
    visits=["init_assess", "1st_repeat_assess", "img", "1st_repeat_img"],
)
met_df /= 24 * 60
build_biomarker(biomarker_df=met_df, time_series=assess_age, biomarker="met")
