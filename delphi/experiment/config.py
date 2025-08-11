from dataclasses import dataclass, field
from typing import Optional

from delphi.log import TrainLogConfig
from delphi.optim import OptimConfig


@dataclass
class TrainBaseConfig:
    ckpt_dir: str = "."
    eval_interval: int = 2000
    eval_iters: int = 200
    eval_only: bool = False  # if True, script exits right after the first eval
    init_from: str = "scratch"

    seed: int = 42
    gradient_accumulation_steps: int = 1  # used to simulate larger batch sizes
    batch_size: int = 128
    # if gradient_accumulation_steps > 1, this is the micro-batch size

    # system
    device: str = "cpu"
    # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: str = "float32"
    # 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile: bool = False  # use PyTorch 2.0 to compile the model to be faster

    distributed_backend: Optional[str] = None

    train_data: dict = field(default_factory=dict)
    val_data: dict = field(default_factory=dict)

    model: dict = field(default_factory=dict)

    optim: OptimConfig = field(default_factory=OptimConfig)

    log: TrainLogConfig = field(default_factory=TrainLogConfig)
