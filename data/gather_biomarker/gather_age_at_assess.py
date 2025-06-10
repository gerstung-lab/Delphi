import os
import warnings

import pandas as pd
from utils import MULTIMODAL_INPUT_DIR, all_ukb_participants

mob_txt = os.path.join(
    MULTIMODAL_INPUT_DIR,
    "general",
    "year_and_month_of_birth.txt",
)
assess_date_txt = os.path.join(
    MULTIMODAL_INPUT_DIR,
    "general",
    "Date_of_attending_assessment_centre_53.txt",
)

mob_df = pd.read_csv(mob_txt, sep="\t", index_col="eid")
mob_df["year_month"] = pd.to_datetime(mob_df["year_month"], format="%Y%m")
assess_date_df = pd.read_csv(assess_date_txt, sep="\t", index_col="f.eid")
assess_date_df["first_visit_date"] = pd.to_datetime(
    assess_date_df["f.53.0.0"], format="%Y-%m-%d"
)
assert (mob_df.index == assess_date_df.index).all()

participants = all_ukb_participants()
if len(participants) != len(mob_df.index):
    missing = set(participants) - set(mob_df.index)
    warnings.warn(f"{len(missing)} participants do not have mob and assessment date")

assess_age_in_days = (assess_date_df["first_visit_date"] - mob_df["year_month"]).dt.days
first_assess_age_df = pd.DataFrame(assess_age_in_days, columns=["age"])
first_assess_age_df.index.name = "pid"
first_assess_age_df.index = first_assess_age_df.index.astype(str)
first_assess_age_df.to_csv(
    os.path.join(MULTIMODAL_INPUT_DIR, "general", "age_at_assess.csv"),
    header=True,
)
