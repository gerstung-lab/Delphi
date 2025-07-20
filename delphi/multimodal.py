from enum import Enum


class Modality(Enum):
    # 0 for padding; 1 for event tokens
    PRS = 2
    WBC = 3
    LIPID = 4
    LFT = 5
    RENAL = 6
    HBA1C = 7
    CRP = 8
    URATE = 9
    CYSC = 10
    APO = 11
    VITD = 12
    DHT = 13
    SHBG = 14
    IGF1 = 15
    NAK = 16
    CREAT = 17
    ALBU = 18
    DIET = 19
    MET = 20
    TELOMERE = 21


def module_name(modality: Modality) -> str:

    module_name = str(modality).split(".")[-1].lower()

    return module_name
