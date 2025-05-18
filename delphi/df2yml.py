import os
import re
from dataclasses import asdict

import pandas as pd
import yaml

from delphi.tokenizer import Disease, TokenizerSchema

tokenizer_df = pd.read_csv(
    "data/ukb_simulated_data/labels.csv", names=["disease"], sep="\t"
)

chapter_df = pd.read_csv(
    "delphi_labels_chapters_colours_icd.csv", quotechar='"', sep=",", engine="python"
)
chapter_df["name"] = chapter_df["name"].str.lower().str.replace(" ", "_")
chapter_df["ICD-10 Chapter (short)"] = (
    chapter_df["ICD-10 Chapter (short)"].str.lower().str.replace(" ", "_")
)
chapter_df.set_index("name", inplace=True)

lifestyle_keywords = ["BMI", "Smoking", "Alcohol"]

disease_mapping = {}
chapters = {chapter: [] for chapter in chapter_df["ICD-10 Chapter (short)"].unique()}
lifestyle_tokens = []
for i, (_, row) in enumerate(tokenizer_df.iterrows(), start=0):
    short_name = row["disease"].lower().replace(" ", "_")
    short_name_no_parentheses = re.sub(r"\(|\)", "", short_name)
    chapter = str(chapter_df.loc[short_name_no_parentheses, "ICD-10 Chapter (short)"])
    chapters[chapter].append(short_name)
    disease_mapping[short_name] = Disease(
        id=i, ukb_name=row["disease"], chapter=chapter
    )
    has_lifestyle_keyword = any(
        keyword in row["disease"] for keyword in lifestyle_keywords
    )
    if has_lifestyle_keyword:
        lifestyle_tokens.append(i)

tokenizer = TokenizerSchema(
    version="0.3", lifestyle=lifestyle_tokens, mapping=disease_mapping
)

with open("data/ukb_tokenizer.yaml", "w") as f:
    yaml.dump(asdict(tokenizer), f, default_flow_style=False, sort_keys=False)

for chapter in chapters.keys():
    with open(os.path.join("config/disease_list", f"{chapter}.yaml"), "w") as f:
        yaml.dump(chapters[chapter], f, default_flow_style=False, sort_keys=False)
