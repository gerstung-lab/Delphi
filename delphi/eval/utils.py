import pandas as pd


def write_auc_results(auc_val: float, n_ctl: int, n_dis: int, csv_path: str) -> None:

    column_types = {
        "age_group": "string",
        "auc": "float32",
        "ctl_counts": "uint32",
        "dis_counts": "uint32",
    }
    df = pd.DataFrame(
        {
            "age_group": ["total"],
            "auc": [auc_val],
            "ctl_counts": [n_ctl],
            "dis_counts": [n_dis],
        }
    ).astype(column_types)
    df.to_csv(
        csv_path,
        index=False,
        float_format="%.3f",
    )
