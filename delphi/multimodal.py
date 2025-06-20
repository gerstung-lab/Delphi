from enum import Enum


class Modality(Enum):
    # 0 for padding; 1 for event tokens
    PRS = 2
    FAMILY_HX = 3
    BLOOD_ALL = 4
    WBC = 5
    LIPID = 6
    LFT = 7
    RENAL = 8
    HBA1C = 9
    CRP = 10
    URATE = 11
    CYSC = 12
    APO = 13
    VITD = 14
    DHT = 15
    SHBG = 16
    IGF1 = 17
    NAK = 18
    CREAT = 19
    ALBU = 20
    MEDS = 21


def module_name(modality: Modality) -> str:

    module_name = str(modality).split(".")[-1].lower()

    return module_name
