import os

from omegaconf import OmegaConf

from delphi.env import DELPHI_CKPT_DIR
from delphi.eval import clock, mimic, ukb


@clock
def main():

    cli_args = OmegaConf.from_cli()
    if hasattr(cli_args, "config"):
        task_args = OmegaConf.load(cli_args.config)
        del cli_args.config
    else:
        task_args = OmegaConf.create(dict())

    ckpt = cli_args.ckpt
    ckpt = os.path.join(DELPHI_CKPT_DIR, ckpt)
    assert os.path.exists(ckpt), f"checkpoint {ckpt} does not exist."
    del cli_args.ckpt

    task = cli_args.task
    del cli_args.task

    if task == "ukb-auc":
        default_args = OmegaConf.structured(ukb.auc.InstantAUCArgs)
        cfg = OmegaConf.merge(default_args, task_args, cli_args)
        cfg = OmegaConf.to_object(cfg)
        ukb.auc.calibrate_auc(task_args=cfg, ckpt=ckpt)
    elif task == "ukb-forecast":
        default_args = OmegaConf.structured(ukb.forecast.ForecastArgs)
        cfg = OmegaConf.merge(default_args, task_args, cli_args)
        cfg = OmegaConf.to_object(cfg)
        ukb.forecast.sample_future(task_args=cfg, ckpt=ckpt)
    elif task == "ukb-loss":
        task_args.update(cli_args)
        ukb.loss.estimate_loss(ckpt=ckpt, **task_args)
    elif task == "mimic-forecast":
        default_args = OmegaConf.structured(mimic.forecast.ForecastArgs)
        cfg = OmegaConf.merge(default_args, task_args, cli_args)
        cfg = OmegaConf.to_object(cfg)
        mimic.forecast.sample_future(task_args=cfg, ckpt=ckpt)
        pass
    elif task == "mimic-drg":
        task_args.update(cli_args)
        mimic.drg.eval(ckpt=ckpt, **task_args)
    elif task == "mimic-sofa":
        task_args.update(cli_args)
        mimic.sofa.drg_classification(ckpt=ckpt, **task_args)
    elif task == "mimic-loss":
        task_args.update(cli_args)
        mimic.loss.estimate_loss(ckpt=ckpt, **task_args)
    else:
        raise ValueError


if __name__ == "__main__":
    main()
