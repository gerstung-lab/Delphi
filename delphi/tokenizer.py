import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Union

import yaml
from dacite import from_dict


class Gender(Enum):
    MALE = "male"
    FEMALE = "female"


class CoreEvents(Enum):
    NO_EVENT = "no_event"
    PADDING = "padding"
    DEATH = "death"


@dataclass
class Disease:
    id: int
    ukb_name: str
    chapter: str


@dataclass
class TokenizerSchema:
    version: str = "0.0"
    lifestyle: list = field(default_factory=list)
    mapping: dict[str, Disease] = field(default_factory=dict)


class Tokenizer:

    def __init__(self, tokenizer_schema: TokenizerSchema):

        self.version = tokenizer_schema.version
        self.mapping = tokenizer_schema.mapping

        core_events = {e.value for e in CoreEvents}
        missing_core_events = core_events - set(self.mapping.keys())
        if missing_core_events:
            raise ValueError(f"missing core events in tokenizer: {missing_core_events}")

        assert (
            Gender.MALE.value in self.mapping.keys()
            and Gender.FEMALE.value in self.mapping.keys()
        )

        self.disease_ids = [disease.id for disease in self.mapping.values()]
        assert len(self.disease_ids) == len(
            set(self.disease_ids)
        ), "disease IDs must be unique"

        # reverse mapping for decoding
        self.id2disease = {disease.id: k for k, disease in self.mapping.items()}

        assert min(self.disease_ids) == 0, "disease IDs must start at 0"
        assert (
            self[CoreEvents.PADDING.value] == 0
        ), "padding token must be the same as the padding key"

        self.vocab_size = len(self.disease_ids)

        self.lifestyle_tokens = tokenizer_schema.lifestyle

    def __getitem__(self, key: str) -> int:
        if key not in self.mapping:
            raise KeyError(f"disease key {key} not found in tokenizer.")

        return self.mapping[key].id

    def encode(self, token: Union[str, list]) -> Union[int, list]:
        if isinstance(token, list):
            for t in token:
                if t not in self.mapping:
                    raise KeyError(f"disease key {t} not found in tokenizer.")
            return [self[t] for t in token]
        else:
            if token not in self.mapping:
                raise KeyError(f"disease key {token} not found in tokenizer.")
            return self[token]

    def name_for_plot(self, key: str) -> str:

        return self.mapping[key].ukb_name

    def decode(self, token: Union[int, list]) -> Union[str, list]:
        if isinstance(token, list):
            for t in token:
                if t not in self.disease_ids:
                    raise KeyError(f"disease ID {t} not found in tokenizer.")
            return [self.id2disease[t] for t in token]
        else:
            if token not in self.disease_ids:
                raise KeyError(f"disease ID {token} not found in tokenizer.")
            return self.id2disease[token]


def load_tokenizer_from_yaml(
    filepath: str,
) -> Tokenizer:
    """
    Load a tokenizer from a yaml file.
    """
    with open(filepath, "r") as f:
        tokenizer_schema = from_dict(
            TokenizerSchema,
            yaml.safe_load(f),
        )

    return Tokenizer(tokenizer_schema)


def load_tokenizer_from_ckpt(
    ckpth_path,
) -> Tokenizer:

    tokenizer_path = os.path.join(ckpth_path, "tokenizer.yaml")
    tokenizer = load_tokenizer_from_yaml(tokenizer_path)

    return tokenizer
