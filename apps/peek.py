import os
from dataclasses import dataclass

from omegaconf import OmegaConf

from apps.interact import unpack
from delphi import DAYS_PER_YEAR
from delphi.data.cohort import build_ukb_cohort
from delphi.data.dataset import UKBDataConfig
from delphi.tokenizer import load_tokenizer_from_yaml


@dataclass
class InspectArgs:
    disease: str = "c25_malignant_neoplasm_of_pancreas"
    disease_t0: float = 0.0
    disease_t1: float = 100.0
    n_samples: int = 10
    sample_t0: float = 0.0
    sample_t1: float = 40.0


def main():

    cli_args = OmegaConf.from_cli()
    default_args = OmegaConf.structured(InspectArgs)
    cfg = OmegaConf.merge(default_args, cli_args)
    # convert structured config back to underlying dataclass
    cfg = OmegaConf.to_object(cfg)

    ukb_data_cfg = UKBDataConfig(
        data_dir="data/ukb_simulated_data",
        memmap_fname="train.bin",
    )

    ukb_cohort = build_ukb_cohort(cfg=ukb_data_cfg)
    tokenizer = load_tokenizer_from_yaml(
        os.path.join(ukb_data_cfg.data_dir, ukb_data_cfg.tokenizer_fname)
    )

    disease_token = tokenizer[cfg.disease]
    time_range = (cfg.disease_t0 * DAYS_PER_YEAR, cfg.disease_t1 * DAYS_PER_YEAR)
    has_token = ukb_cohort.has_token(disease_token, time_range=time_range)

    has_any_in_time_range = ukb_cohort.has_any_token(
        time_range=(cfg.sample_t0 * DAYS_PER_YEAR, cfg.sample_t1 * DAYS_PER_YEAR)
    )
    cohort_with_disease = ukb_cohort[has_token & has_any_in_time_range]
    sample_trajectories = cohort_with_disease.sample_trajectory(
        n_samples=cfg.n_samples,
        time_range=(cfg.sample_t0 * DAYS_PER_YEAR, cfg.sample_t1 * DAYS_PER_YEAR),
    )

    for tokens, timesteps in sample_trajectories:

        tokens = tokens.tolist()
        events = tokenizer.decode(tokens)
        timesteps = timesteps.astype("float") / DAYS_PER_YEAR
        timesteps = timesteps.tolist()

        for event, time in zip(events, timesteps):
            print(f"{event}, {time:.2f}")


if __name__ == "__main__":
    main()
