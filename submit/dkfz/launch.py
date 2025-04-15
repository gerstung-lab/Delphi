import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional

from omegaconf import OmegaConf


@dataclass
class RunConfig:
    name: str = "debug"
    script: str = "apps/train.py"
    script_args: str = ""
    memory: int = 16
    gpu: bool = True
    gpu_num: int = 1
    j_exclusive: bool = True
    gpu_mem: int = 10
    queue: str = "gpu-debian"
    stdout: Optional[str] = None
    stderr: Optional[str] = None


@dataclass
class ExperimentConfig:
    submit_script: str = "./submit/dkfz/submit.sh"
    runs: list[RunConfig] = field(default_factory=list)


def validate_run_config(run_config: RunConfig):
    pass


def parse_memory_config(run_cfg: RunConfig) -> str:
    return f"rusage[mem={run_cfg.memory}GB]"


def parse_gpu_config(run_cfg: RunConfig) -> str:

    if run_cfg.gpu:
        num = f"num={run_cfg.gpu_num}:"
        j_exclusive = "j_exclusive=yes:" if run_cfg.j_exclusive else ""
        gmem = f"gmem={run_cfg.gpu_mem}G"
        return f"{num}{j_exclusive}{gmem}"
    else:
        return ""


def main():

    cli_cfg = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_cfg.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_cfg.config

    default_cfg = OmegaConf.structured(ExperimentConfig)
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_cfg)
    cfg = OmegaConf.to_object(cfg)
    print(cfg)

    submit_script = cfg.submit_script

    for run_cfg in cfg.runs:
        validate_run_config(run_cfg)

        command = "bsub"

        memory_args = parse_memory_config(run_cfg)
        command += f" -R {memory_args}"

        if run_cfg.gpu:
            gpu_args = parse_gpu_config(run_cfg)
            command += f" -gpu {gpu_args}"

        if run_cfg.stdout:
            os.makedirs(os.path.dirname(run_cfg.stdout), exist_ok=True)
            command += f" -o {run_cfg.stdout}"

        if run_cfg.stderr:
            os.makedirs(os.path.dirname(run_cfg.stderr), exist_ok=True)
            command += f" -e {run_cfg.stderr}"

        command += f" -q {run_cfg.queue}"

        command += (
            f' /bin/bash -l -c "{submit_script} {run_cfg.script} {run_cfg.script_args}"'
        )
        print(f"{run_cfg.name}")
        print(f"{command}")

        try:
            result = subprocess.run(command, check=True, shell=True)
            sys.exit(result.returncode)
        except subprocess.CalledProcessError as e:
            print(f"Script exited with error: {e.returncode}")
            sys.exit(e.returncode)


if __name__ == "__main__":
    main()
