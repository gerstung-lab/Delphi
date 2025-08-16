from pathlib import Path

import torch as th

from .base import InferenceDataset
from .constants import SpecialToken as ST
from .hospital_mortality import HospitalMortalityBase


class DrgPredictionDataset(InferenceDataset):
    """Produces timelines that end at the last ICD-related token associated with the patient's
    hospital stay.

    The target is the DRG token. Omits cases where the code is DRG//UNKNOWN.
    """

    def __init__(self, input_dir: str | Path, n_positions: int = 2048, **kwargs):
        super().__init__(input_dir, n_positions, **kwargs)
        self.stop_stokens = [
            stoken for stoken in self.vocab if stoken.startswith("DRG//")
        ]
        self.drg_indices = self._get_indices_of_stokens(
            set(self.stop_stokens).difference(["DRG//UNKNOWN"])
        )

    def __len__(self) -> int:
        return len(self.drg_indices)

    def __getitem__(self, idx: int) -> tuple[th.Tensor, dict]:
        drg_idx = self.drg_indices[idx]
        data_idx = drg_idx - 1
        return super().__getitem__(data_idx), {
            #! shift by 2
            "expected": self.vocab.decode(self.tokens[drg_idx] + 2),
            "hadm_id": self._get_hadm_id(drg_idx),
            "true_token_dist": 1,
            "true_token_time": 0,
            "patient_id": self.patient_id_at_idx[drg_idx].item(),
            "data_idx": data_idx.item(),
        }


class SofaPredictionDataset(InferenceDataset):
    """Produces timelines that end on the ICU_ADMISSION token.

    The target is the quantile token of the sofa score. The 'icu_stay_id' can be used to get the
    real groundtruth later.
    """

    def __init__(self, input_dir: str | Path, n_positions: int = 2048, **kwargs):
        super().__init__(input_dir, n_positions, **kwargs)
        self.stop_stokens = self.vocab.quantile_stokens
        self.icu_adm_indices = self._get_indices_of_stokens(ST.ICU_ADMISSION)

    def __len__(self) -> int:
        return len(self.icu_adm_indices)

    def __getitem__(self, idx: int) -> tuple[th.Tensor, dict]:
        icu_adm_idx = self.icu_adm_indices[idx]
        sofa_token = self.tokens[icu_adm_idx + 3]

        return super().__getitem__(icu_adm_idx), {
            #! shift by 2
            "expected": self.vocab.decode(sofa_token + 2),
            "true_token_dist": 3,
            "icu_stay_id": self._get_icu_stay_id(icu_adm_idx),
            "patient_id": self.patient_id_at_idx[icu_adm_idx].item(),
            "data_idx": icu_adm_idx.item(),
        }


class ICUAdmissionDataset(InferenceDataset):
    """Generates patient timelines ending at the HOSPITAL ADMISSION token. The primary target is to
    predict whether the patient was admitted to the ICU.

    Note:
    - Each death occurring outside the ICU is treated as a positive case, based on the assumption
      that the patient should have been admitted to the ICU.
    """

    def __init__(self, input_dir: str | Path, n_positions: int = 2048, **kwargs):
        super().__init__(input_dir, n_positions, **kwargs)
        self.stop_stokens = [ST.ICU_ADMISSION, ST.DISCHARGE] + self.stop_stokens
        self.adm_indices = self._get_indices_of_stokens(ST.ADMISSION)

        icu_adm_or_dc_or_dth_indices = self._get_indices_of_stokens(
            [ST.DISCHARGE, ST.ICU_ADMISSION, ST.DEATH]
        )
        self.outcome_indices = self._match(
            icu_adm_or_dc_or_dth_indices, self.adm_indices
        )

    def __len__(self) -> int:
        return len(self.adm_indices)

    def __getitem__(self, idx) -> tuple[th.Tensor, dict]:
        start_idx = self.adm_indices[idx] + 2
        outcome_idx = self.outcome_indices[idx]

        return super().__getitem__(start_idx), {
            #! shift by 2
            "expected": self.vocab.decode(self.tokens[outcome_idx] + 2),
            "true_token_dist": (outcome_idx - start_idx).item(),
            "true_token_time": (self.times[outcome_idx] - self.times[start_idx]).item(),
            "icu_stay_id": self._get_icu_stay_id(outcome_idx),
            "patient_id": self.patient_id_at_idx[start_idx].item(),
            "prediction_time": self.times[start_idx].item(),
            "data_idx": start_idx.item(),
        }


class ICUMortalityDataset(HospitalMortalityBase):
    def __init__(self, input_dir: str | Path, n_positions: int = 2048, **kwargs):
        super().__init__(
            input_dir=input_dir,
            adm_stoken=ST.ICU_ADMISSION,
            adm_offset=1,
            dc_stoken=ST.ICU_DISCHARGE,
            n_positions=n_positions,
            **kwargs,
        )


class ICUReadmissionDataset(InferenceDataset):
    """To talk about ICU readmission, there has to be at least one ICU stay within a hospital
    stay."""

    def __init__(self, input_dir: str | Path, n_positions: int = 2048, **kwargs):
        super().__init__(input_dir, n_positions, **kwargs)
        self.stop_stokens = [ST.ICU_ADMISSION, ST.DISCHARGE] + self.stop_stokens
        adm_indices = self._get_indices_of_stokens(ST.ADMISSION)

        # hospital stays with at least one ICU stay
        icu_dc_indices = self._get_indices_of_stokens(ST.ICU_DISCHARGE)
        icu_dc_or_dc_indices = self._get_indices_of_stokens(
            [ST.ICU_DISCHARGE, ST.DISCHARGE, ST.DEATH]
        )
        adm_icu_dc_or_dc_indices = self._match(icu_dc_or_dc_indices, adm_indices)

        has_icu_stay = th.isin(adm_icu_dc_or_dc_indices, icu_dc_indices)
        self.icu_dc_indices = adm_icu_dc_or_dc_indices[has_icu_stay]

        icu_adm_or_dc_indices = self._get_indices_of_stokens(self.stop_stokens)
        self.outcome_indices = self._match(icu_adm_or_dc_indices, self.icu_dc_indices)

    def __len__(self) -> int:
        return len(self.icu_dc_indices)

    def __getitem__(self, idx) -> tuple[th.Tensor, dict]:
        icu_dc_idx = self.icu_dc_indices[idx]
        outcome_idx = self.outcome_indices[idx]

        return super().__getitem__(icu_dc_idx), {
            #! shift by 2
            "expected": self.vocab.decode(self.tokens[outcome_idx] + 2),
            "true_token_dist": (outcome_idx - icu_dc_idx).item(),
            "true_token_time": (
                self.times[outcome_idx] - self.times[icu_dc_idx]
            ).item(),
            "icu_stay_id_start": self._get_icu_stay_id(icu_dc_idx),
            "icu_stay_id_outcome": self._get_icu_stay_id(outcome_idx),
            "patient_id": self.patient_id_at_idx[icu_dc_idx].item(),
            "prediction_time": self.times[icu_dc_idx].item(),
            "data_idx": icu_dc_idx.item(),
        }
