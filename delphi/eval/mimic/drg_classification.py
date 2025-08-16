from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from delphi.data.mimic import DrgPredictionDataset
from delphi.data.utils import eval_iter, move_batch_to_device
from delphi.env import DELPHI_DATA_DIR
from delphi.eval import eval_task
from delphi.experiment.train import load_ckpt


@dataclass
class DRGClassificationArgs:
    data: dict = field(default_factory=dict)
    subsample: Optional[int] = None
    batch_size: int = 128
    device: str = "cuda"


@eval_task.register
def drg_classification(
    task_args: DRGClassificationArgs, task_name: str, ckpt: str
) -> None:

    model, _, tokenizer = load_ckpt(ckpt)
    eval_ds = DrgPredictionDataset(
        input_dir=Path(DELPHI_DATA_DIR) / "mimic" / "test",
        n_positions=model.config.block_size,
        sep_time_tokens=(model.model_type != "ethos"),
    )
