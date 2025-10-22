from dataclasses import dataclass, field

from omegaconf import OmegaConf

from delphi import distributed
from delphi.data.ukb import MultimodalUKBDataset
from delphi.experiment import BaseTrainer, TrainBaseConfig
from delphi.model.multimodal import DelphiM4, DelphiM4Config


@dataclass
class TrainConfig(TrainBaseConfig):
    train_subject_list: str = "participants/train_fold.bin"
    val_subject_list: str = "participants/val_fold.bin"
    model: DelphiM4Config = field(default_factory=DelphiM4Config)
    biomarkers: None | dict[str, int] = None
    expansion_packs: None | list[str] = None


def train(cfg: TrainConfig):

    biomarkers = list(cfg.biomarkers.keys()) if cfg.biomarkers is not None else None
    train_ds = MultimodalUKBDataset(
        biomarkers=biomarkers,
        expansion_packs=cfg.expansion_packs,
        crop_mode="left",
        subject_list=cfg.train_subject_list,
        block_size=cfg.model.block_size,
    )
    val_ds = MultimodalUKBDataset(
        perturb=False,
        biomarkers=biomarkers,
        expansion_packs=cfg.expansion_packs,
        crop_mode="left",
        subject_list=cfg.val_subject_list,
        block_size=cfg.model.block_size,
    )

    cfg.model.vocab_size = train_ds.vocab_size
    cfg.model.ignore_tokens = list(
        set(cfg.model.ignore_tokens).union(train_ds.expansion_tokens)
    )
    if cfg.biomarkers is not None:
        for biomarker, n_features in cfg.biomarkers.items():
            cfg.model.biomarkers[biomarker] = {
                "projector": "linear",
                "input_size": n_features,
            }
    model = DelphiM4(cfg.model)

    backend = distributed.make_backend_from_args(cfg)
    trainer = BaseTrainer(
        cfg=cfg,
        backend=backend,
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
    )
    trainer.train()
    backend.finalize()


def main():

    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    del cli_args.config

    default_cfg = OmegaConf.structured(TrainConfig())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    train(cfg)  # type: ignore


if __name__ == "__main__":
    main()
