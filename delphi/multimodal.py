from enum import Enum


class Modality(Enum):
    # 0 for padding; 1 for event tokens
    PRS = 2
    BLOOD_ALL = 3
    WBC = 4
    LIPID = 5
    LFT = 6
    RENAL = 7
    HBA1C = 8
    CRP = 9
    URATE = 10
    CYSC = 11
    APO = 12
    VITD = 13
    DHT = 14
    SHBG = 15
    IGF1 = 16
    NAK = 17
    CREAT = 18
    ALBU = 19
    DIET = 20
    MET = 21
    TELOMERE = 22


def module_name(modality: Modality) -> str:

    module_name = str(modality).split(".")[-1].lower()

    return module_name
