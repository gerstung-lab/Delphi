import os
import time
from dataclasses import dataclass

import numpy as np
import wandb


@dataclass
class GenLogConfig:
    save_tokens: bool = True
    save_logits: bool = False
    flush_interval: int = 10
    wandb_log: bool = True
    wandb_project: str = "delphi"
    wandb_run_name: str = "gen" + str(time.time())


class GenLogger:
    def __init__(
        self,
        cfg: GenLogConfig,
        dump_dir: str,
    ):
        self.cfg = cfg
        self.dump_dir = dump_dir

        self.wandb = self.cfg.wandb_log
        if self.wandb:
            wandb.init(
                project=self.cfg.wandb_project,
                name=self.cfg.wandb_run_name,
            )

        self.save_tokens = self.cfg.save_tokens
        self.save_logits = self.cfg.save_logits
        if self.save_logits:
            assert self.save_tokens
        self.flush_freq = self.cfg.flush_interval

        self.step = 0
        self.offset = 0

    def init_memmaps(self, n_max_token: int, n_vocab: int):

        if self.save_tokens:
            self.gen_bin = np.memmap(
                os.path.join(self.dump_dir, "gen.bin"),
                dtype=np.uint32,
                mode="w+",
                shape=(int(n_max_token), 3),
            )
            if self.save_logits:
                self.logits_bin = np.memmap(
                    os.path.join(self.dump_dir, "logits.bin"),
                    dtype=np.float16,
                    mode="w+",
                    shape=(int(n_max_token), n_vocab),
                )

    def flush_memmaps(self):
        if self.save_tokens:
            self.gen_bin.flush()
            if self.save_logits:
                self.logits_bin.flush()

    def write_memmaps(
        self,
        participants: np.ndarray,
        tokens: np.ndarray,
        timesteps: np.ndarray,
        logits: np.ndarray,
    ):

        if not hasattr(self, "gen_bin"):
            raise ValueError("memmaps not initialized; call init_memmaps first")

        if self.save_tokens:
            sub_idx, pos_idx = np.nonzero(tokens > 0)
            packed_batch = np.stack(
                (
                    participants[sub_idx],
                    tokens[sub_idx, pos_idx],
                    timesteps[sub_idx, pos_idx],
                ),
                axis=-1,
            )
            token_n = packed_batch.shape[0]
            self.gen_bin[self.offset : self.offset + token_n, :] = packed_batch
            if self.save_logits:
                packed_logits = logits[..., sub_idx, pos_idx, :]
                self.logits_bin[self.offset : self.offset + token_n, :] = packed_logits

        if self.step % self.flush_freq == 0:
            self.flush_memmaps()

        self.step += 1
        self.offset += token_n

    def close(self):
        if self.save_tokens:
            self.gen_bin.flush()
