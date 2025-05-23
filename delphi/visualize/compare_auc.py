import os
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from delphi.eval import eval_task
from delphi.eval.auc import parse_diseases


@dataclass
class CompareAUCArgs:
    disease_lst: str = ""
    ctrl_auc_dir: str = ""


def fetch_auc(disease_auc_dir: str, gender: Literal["male", "female", "either"]):
    auc_df = pd.read_csv(
        os.path.join(disease_auc_dir, f"{gender}_0.1.csv"), index_col=0
    )
    return auc_df.loc["total", "auc"]


def scatter_plot(
    ckpt_auc: list,
    ctrl_auc: list,
    savefig: str,
) -> None:

    fig, ax = plt.subplots()
    sns.scatterplot(
        x=ctrl_auc,
        y=ckpt_auc,
        ax=ax,
        color="blue",
        edgecolor="black",
        linewidth=0.5,
        s=100,
        alpha=0.7,
        marker="o",
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
    plt.savefig(
        savefig,
        dpi=300,
        bbox_inches="tight",
    )


@eval_task.register
def compare_auc(
    task_args: CompareAUCArgs, task_name: str, task_input: str, ckpt: str, **kwargs
):
    diseases = parse_diseases(task_args.disease_lst)

    ckpt_auc_dir = os.path.join(ckpt, task_input)
    assert os.path.exists(
        ckpt_auc_dir
    ), f"ckpt auc directory {ckpt_auc_dir} does not exist."
    assert os.path.exists(
        task_args.ctrl_auc_dir
    ), f"ctrl auc directory {task_args.ctrl_auc_dir} does not exist."

    ckpt_auc = []
    ctrl_auc = []

    for disease in diseases:

        ckpt_disease_auc_dir = os.path.join(ckpt_auc_dir, disease)
        assert os.path.exists(
            ckpt_disease_auc_dir
        ), f"auc for {disease} not found in {ckpt_auc_dir}"
        ckpt_auc.append(fetch_auc(ckpt_disease_auc_dir, "either"))

        ctrl_disease_auc_dir = os.path.join(task_args.ctrl_auc_dir, disease)
        assert os.path.exists(
            ctrl_disease_auc_dir
        ), f"auc for {disease} not found in {task_args.ctrl_auc_dir}"
        ctrl_auc.append(fetch_auc(ctrl_disease_auc_dir, "either"))

    scatter_plot(
        ckpt_auc=ckpt_auc,
        ctrl_auc=ctrl_auc,
        savefig=os.path.join(ckpt_auc_dir, f"{task_name}.png"),
    )
