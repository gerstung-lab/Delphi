import os
from enum import Enum
from typing import Union

import yaml

MALE = "male"
FEMALE = "female"
NO_EVENT = "no_event"
PADDING = "padding"


class Tokenizer:

    def __init__(self, name2id: dict) -> None:

        self.name2id = name2id
        assert len(self.name2id.values()) == len(
            set(self.name2id.values())
        ), "tokens must be unique"

        self.id2name = {v: k for k, v in self.name2id.items()}
        self.vocab_size = len(self.name2id)

    def __repr__(self) -> str:
        return f"Tokenizer(vocab_size={self.vocab_size}, name2id={self.name2id})"

    def __getitem__(self, key: str) -> int:
        if key not in self.name2id:
            raise KeyError(f"disease key {key} not found in tokenizer.")

        return self.name2id[key]

    def to_dict(self):
        return self.name2id

    def encode(self, token: Union[str, list]) -> Union[int, list]:
        if isinstance(token, list):
            for t in token:
                if t not in self.name2id:
                    raise KeyError(f"disease key {t} not found in tokenizer.")
            return [self[t] for t in token]
        else:
            if token not in self.name2id:
                raise KeyError(f"disease key {token} not found in tokenizer.")
            return self[token]

    def decode(self, token: Union[int, list]) -> Union[str, list]:
        if isinstance(token, list):
            for t in token:
                if t not in self.id2name.keys():
                    raise KeyError(f"disease ID {t} not found in tokenizer.")
            return [self.id2name[t] for t in token]
        else:
            if token not in self.id2name.keys():
                raise KeyError(f"disease ID {token} not found in tokenizer.")
            return self.id2name[token]


def load_tokenizer_from_yaml(
    filepath: str | os.PathLike,
) -> Tokenizer:
    assert os.path.exists(filepath), f"tokenizer file {filepath} does not exist"
    with open(filepath, "r") as f:
        name2id = yaml.safe_load(f)

    return Tokenizer(name2id=name2id)


def update_tokenizer(base_tokenizer: dict, add_tokenizer: dict) -> tuple[dict, int]:

    assert min(base_tokenizer.values()) == 0, "base tokenizer must start with 0"
    assert min(add_tokenizer.values()) == 1, "additional tokenizer must start with 1"
    offset = len(base_tokenizer) - 1
    for key, value in add_tokenizer.items():
        if key not in base_tokenizer:
            base_tokenizer[key] = value + offset
        else:
            raise ValueError(f"{key} already exists in base tokenizer")
    return base_tokenizer, offset
