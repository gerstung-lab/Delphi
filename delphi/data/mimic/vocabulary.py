import functools
import json
import operator
from collections.abc import Sequence
from datetime import timedelta
from enum import StrEnum
from pathlib import Path
from typing import Optional, TypeVar

import polars as pl
import torch as th
from loguru import logger

T = TypeVar("T", bound="Vocabulary")


class Vocabulary:
    def __init__(
        self,
        vocab: Optional[Sequence[str]] = None,
        interval_estimates: Optional[dict[str, dict]] = None,
    ):
        if vocab is None:
            vocab = []
        self.stoi = {word: i for i, word in enumerate(vocab)}
        self._itos = None
        self._interval_estimates = interval_estimates

    @classmethod
    @functools.lru_cache(maxsize=1)
    def from_path(cls: type[T], fp: str | Path) -> T:
        vocab_dir = Path(fp)
        if not vocab_dir.is_dir():
            raise ValueError(f"Expected a directory, got {vocab_dir}")

        vocab_path = list(vocab_dir.glob("vocab_t*.csv"))
        if len(vocab_path) == 0:
            raise FileNotFoundError(
                f"Searching for a vocab file (vocab_t*.csv) in {vocab_dir} yielded no results"
            )
        if len(vocab_path) > 1:
            raise ValueError(
                f"Searching for a vocab file (vocab_t*.csv) in {vocab_dir} yielded more than one "
                f"result: {vocab_path}"
            )
        vocab = pl.read_csv(vocab_path[0], has_header=False).to_series(0).to_list()

        interval_estimates_fp = vocab_dir / "interval_estimates.json"
        if interval_estimates_fp.is_file():
            with interval_estimates_fp.open("r") as f:
                interval_estimates = json.load(f)
        else:
            logger.warning(f"Interval estimates not found in {interval_estimates_fp}")
            interval_estimates = None

        return cls(vocab, interval_estimates)

    def dump(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        if not output_dir.is_dir():
            raise ValueError(f"Expected a directory, got {output_dir}")

        vocab = pl.DataFrame(list(self.stoi.keys()))
        vocab.write_csv(output_dir / f"vocab_t{len(self)}.csv", include_header=False)

    def __len__(self):
        return len(self.stoi)

    def __iter__(self):
        return iter(self.stoi.keys())

    def add_words(self, words: str | Sequence[str]) -> None:
        if isinstance(words, str):
            words = [words]
        for word in words:
            if not isinstance(word, str):
                raise ValueError(f"Expected str, {word} is '{type(word)}'")
            if word not in self.stoi:
                self.stoi[word] = len(self.stoi)

    def encode(self, codes: str | Sequence[str]) -> list[int] | int:
        if isinstance(codes, th.Tensor):
            codes = codes.tolist()
        if isinstance(codes, str):
            return self.stoi[codes]
        return [self.stoi[code] for code in codes]

    def decode(self, tokens: int | Sequence[int]) -> list[str] | str:
        if isinstance(tokens, th.Tensor):
            tokens = tokens.tolist()
        if isinstance(tokens, int):
            return self.itos[tokens]
        return [self.itos[token] for token in tokens]

    @property
    def itos(self) -> dict[int, str]:
        if self._itos is None or len(self._itos) != len(self.stoi):
            self._itos = {v: k for k, v in self.stoi.items()}
        return self._itos

    @property
    def interval_estimates(self) -> dict[str, dict]:
        if self._interval_estimates is None:
            raise ValueError(
                "The interval estimates file not found during the initialization."
            )
        return self._interval_estimates

    @property
    def quantile_stokens(self) -> list[str]:
        return [t for t in self.stoi.keys() if t.startswith("Q") and t[1:].isdigit()]

    @property
    def time_interval_stokens(self) -> list[str]:
        return list(self.interval_estimates["mean"].keys())

    def get_timeline_total_time(
        self,
        timeline: Sequence[int | str],
        stat: str = "mean",
        input_str: bool = False,
    ) -> timedelta:
        if not input_str:
            timeline = (self.decode(t) for t in timeline)  # type: ignore

        interval_estimates = self.interval_estimates[stat]
        return functools.reduce(
            operator.add,
            (
                timedelta(microseconds=interval_estimates[t])
                for t in timeline
                if t in interval_estimates
            ),
            timedelta(0),
        )
