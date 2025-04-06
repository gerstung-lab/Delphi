import argparse
import logging
from typing import Type, TypeVar, Union

from omegaconf import DictConfig, ListConfig, OmegaConf

T = TypeVar("T")


def dataclass_from_dict(
    cls: Type[T], data: Union[DictConfig, dict], strict: bool = True
) -> T:
    """
    Converts a dictionary to a dataclass instance, recursively for nested structures.
    """
    base = OmegaConf.structured(cls())
    OmegaConf.set_struct(base, strict)
    override = OmegaConf.create(data)
    return OmegaConf.to_object(OmegaConf.merge(base, override))  # type: ignore
