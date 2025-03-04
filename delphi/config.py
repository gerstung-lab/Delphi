import argparse
import logging
from omegaconf import OmegaConf, DictConfig, ListConfig
from typing import Type, TypeVar

T = TypeVar("T")

def dataclass_from_dict(cls: Type[T], data: dict, strict: bool = True) -> T:
    """
    Converts a dictionary to a dataclass instance, recursively for nested structures.
    """
    base = OmegaConf.structured(cls())
    OmegaConf.set_struct(base, strict)
    override = OmegaConf.create(data)
    return OmegaConf.to_object(OmegaConf.merge(base, override))