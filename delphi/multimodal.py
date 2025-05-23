from enum import Enum

import torch


class Modality(Enum):
    PRS = 2
    FAMILY_HX = 3


modality_X_dtype = {
    Modality.PRS: torch.float32,
    Modality.FAMILY_HX: torch.long,
}


def module_name(modality: Modality) -> str:

    module_name = str(modality).split(".")[-1].lower()

    return module_name
