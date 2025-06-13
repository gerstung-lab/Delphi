import json
import os
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

from delphi.env import DELPHI_CKPT_DIR
from delphi.eval import eval_task
from delphi.tokenizer import Gender


@dataclass
class CompareAUCArgs:
    disease_lst: str = ""
    male_only_disease_lst: Optional[str] = None
    female_only_disease_lst: Optional[str] = None
    baseline_auc_json: str = ""


def scatter_plot(
    ckpt_auc: list,
    ctrl_auc: list,
    n_dis: list,
) -> None:

    fig, ax = plt.subplots()

    color_max = np.log(np.array(n_dis)).max()
    color_values = [
        np.log(n) / color_max for n in n_dis
    ]  # Normalize n_dis for color mapping
    ax.scatter(
        x=ctrl_auc,
        y=ckpt_auc,
        c=color_values,  # Use color_values for color mapping
        cmap="Blues",  # Color map (darker = higher values)
        edgecolor="black",
        linewidth=0.5,
        s=100,
        alpha=0.7,
        marker="o",
        vmin=0,  # Ensure 0 maps to lightest color
        vmax=1,  # Ensure 1 maps to darkest color
    )
    sns.lineplot(
        x=[0, 1],
        y=[0, 1],
        ax=ax,
        color="red",
        linestyle="--",
        linewidth=1,
    )
    plt.xlabel("baseline")

    delta_mu = np.nanmean(np.array(ckpt_auc) - np.array(ctrl_auc))
    delta_std = np.nanstd(np.array(ckpt_auc) - np.array(ctrl_auc))
    n_improved = np.sum(np.array(ckpt_auc) > np.array(ctrl_auc))
    plt.title(
        f"avg delta: {delta_mu:.3f}({delta_std:.3f}), n improved: {n_improved}/{len(ckpt_auc)}"
    )


@eval_task.register
def compare_auc(
    task_args: CompareAUCArgs, task_name: str, task_input: str, ckpt: str, **kwargs
):

    with open(task_args.disease_lst, "r") as f:
        diseases = yaml.safe_load(f)
    male_only_diseases = []
    if task_args.male_only_disease_lst is not None:
        with open(task_args.male_only_disease_lst, "r") as f:
            male_only_diseases = yaml.safe_load(f)
    female_only_diseases = []
    if task_args.female_only_disease_lst is not None:
        with open(task_args.female_only_disease_lst, "r") as f:
            female_only_diseases = yaml.safe_load(f)

    with open(os.path.join(DELPHI_CKPT_DIR, ckpt, task_input), "r") as f:
        auc_logbook = json.load(f)

    with open(task_args.baseline_auc_json, "r") as f:
        bl_auc_logbook = json.load(f)

    auc_lst = {}
    bl_auc_lst = {}
    n_dis_list = {}

    for gender in Gender:
        gender = gender.value
        auc_lst[gender] = []
        bl_auc_lst[gender] = []
        for disease in diseases:
            if disease in male_only_diseases:
                gender = "male"
            if disease in female_only_diseases:
                gender = "female"

            auc_lst[gender].append(bl_auc_logbook[disease][gender]["total"]["auc"])
            n_dis_list[gender].append(
                bl_auc_logbook[disease][gender]["total"]["dis_count"]
            )
            bl_auc_lst[gender].append(auc_logbook[disease][gender]["total"]["auc"])

            scatter_plot(
                ckpt_auc=auc_lst,
                ctrl_auc=bl_auc_lst,
                n_dis=n_dis_list,
            )

            plt.savefig(
                os.path.join(os.path.dirname(task_input), f"{task_name}.png"),
                dpi=300,
                bbox_inches="tight",
            )
