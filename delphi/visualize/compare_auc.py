import glob
import os
from dataclasses import asdict, dataclass
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from delphi.env import DELPHI_CKPT_DIR
from delphi.eval import eval_task
from delphi.eval.auc import parse_diseases


@dataclass
class CompareAUCArgs:
    disease_lst: str = ""
    gender: str = "either"
    male_only_disease_lst: Optional[str] = None
    female_only_disease_lst: Optional[str] = None
    ctrl_ckpt: str = ""
    ctrl_task_input: str = ""
    plot_name: str = ""


def fetch_auc(
    disease_auc_dir: str, gender: Literal["male", "female", "either"]
) -> tuple[float, int]:

    # for backward compatibility, some .csv files were name as f"{gender}_0.1.csv"
    pattern = os.path.join(glob.escape(disease_auc_dir), f"{gender}*.csv")
    matching_files = glob.glob(pattern)
    if not matching_files:
        raise FileNotFoundError(f"no file found matching pattern: {pattern}")
    if len(matching_files) > 1:
        raise RuntimeError(f"multiple files found for pattern: {pattern}")

    auc_df = pd.read_csv(matching_files[0], index_col=0)

    return auc_df.loc["total", "auc"], auc_df.loc["total", "dis_counts"]  # type: ignore


def scatter_plot(
    ckpt_auc: list,
    ctrl_auc: list,
    n_dis: list,
    savefig: str,
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
    plt.savefig(
        savefig,
        dpi=300,
        bbox_inches="tight",
    )


@eval_task.register
def compare_auc(
    task_args: CompareAUCArgs, task_name: str, task_input: str, ckpt: str, **kwargs
):

    task_dump_dir = os.path.join(ckpt, task_input, task_name)
    os.makedirs(task_dump_dir, exist_ok=True)
    with open(os.path.join(task_dump_dir, "config.yaml"), "w") as f:
        yaml.dump(asdict(task_args), f, default_flow_style=False, sort_keys=False)

    diseases = parse_diseases(task_args.disease_lst)

    ckpt_auc_dir = os.path.join(ckpt, task_input)
    assert os.path.exists(
        ckpt_auc_dir
    ), f"ckpt auc directory {ckpt_auc_dir} does not exist."

    ctrl_ckpt = os.path.join(DELPHI_CKPT_DIR, task_args.ctrl_ckpt)
    assert os.path.exists(ctrl_ckpt), f"control ckpt {ctrl_ckpt} does not exist."
    ctrl_ckpt_auc_dir = os.path.join(ctrl_ckpt, task_args.ctrl_task_input)
    assert os.path.exists(
        ctrl_ckpt_auc_dir
    ), f"ctrl auc directory {ctrl_ckpt_auc_dir} does not exist."

    male_only_diseases = []
    if task_args.male_only_disease_lst is not None:
        male_only_diseases = parse_diseases(task_args.male_only_disease_lst)
    female_only_diseases = []
    if task_args.female_only_disease_lst is not None:
        female_only_diseases = parse_diseases(task_args.female_only_disease_lst)

    ckpt_auc_list = []
    ctrl_auc_list = []
    n_dis_list = []

    for disease in diseases:

        gender = task_args.gender
        if disease in male_only_diseases:
            gender = "male"
        if disease in female_only_diseases:
            gender = "female"

        ckpt_disease_auc_dir = os.path.join(ckpt_auc_dir, disease)
        assert os.path.exists(
            ckpt_disease_auc_dir
        ), f"auc for {disease} not found in {ckpt_auc_dir}"

        ckpt_auc, ckpt_n_dis = fetch_auc(ckpt_disease_auc_dir, gender)  # type: ignore
        ckpt_auc_list.append(ckpt_auc)
        n_dis_list.append(ckpt_n_dis)

        ctrl_disease_auc_dir = os.path.join(ctrl_ckpt_auc_dir, disease)
        assert os.path.exists(
            ctrl_disease_auc_dir
        ), f"auc for {disease} not found in {ctrl_ckpt_auc_dir}"
        ctrl_auc, _ = fetch_auc(ctrl_disease_auc_dir, gender)  # type: ignore
        ctrl_auc_list.append(ctrl_auc)

    scatter_plot(
        ckpt_auc=ckpt_auc_list,
        ctrl_auc=ctrl_auc_list,
        n_dis=n_dis_list,
        savefig=os.path.join(task_dump_dir, f"{task_args.plot_name}.png"),
    )
