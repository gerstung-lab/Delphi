import math
import os
from contextlib import contextmanager

import torch
from torch.distributed import destroy_process_group, get_world_size, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from .backend import DistributedBackend


class DataParallelDistributedBackend(DistributedBackend):

    def __init__(self, args):
        self.rank = int(os.environ.get("RANK", -1))
        assert self.rank != -1, "DDP backend can not be used without rank"
        assert "cuda" in args.device, "DDP backend can not be used on non-CUDA devices"
        init_process_group(backend=args.distributed_backend)
        self.local_rank = int(os.environ["LOCAL_RANK"])

    def get_adjusted_args_for_process(self, args):
        effective_batch_size = args.batch_size * args.gradient_accumulation_steps
        world_size = self.get_world_size()
        if effective_batch_size % world_size != 0:
            raise ValueError(
                f"Effective batch size "
                "{effective_batch_size} is not divisible "
                "by the world size {world_size}."
            )
        acc_steps_div = math.gcd(args.gradient_accumulation_steps, world_size)
        args.gradient_accumulation_steps = (
            args.gradient_accumulation_steps // acc_steps_div
        )
        args.batch_size = args.batch_size // (world_size // acc_steps_div)
        args.device = f"cuda:{self.local_rank}"
        args.seed = args.seed + self.local_rank
        # args.data_seed = args.data_seed
        return args

    def transform_model(self, model):
        return DDP(model, device_ids=[self.local_rank])

    @contextmanager
    def get_context_for_microstep_forward(
        self,
        model: torch.nn.Module,
        microstep_idx: int,
        gradient_accumulation_steps: int,
    ):
        model.require_backward_grad_sync = (
            microstep_idx == gradient_accumulation_steps - 1
        )
        yield

    def is_master_process(self) -> bool:
        return self.rank == 0

    def get_raw_model(self, model: torch.nn.Module):
        return model.module

    def translate_model_parameter_name_for_node(self, parameter_name):
        return [f"module.{parameter_name}"]

    def get_world_size(self):
        return get_world_size()

    def finalize(self):
        destroy_process_group()
