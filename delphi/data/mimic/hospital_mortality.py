import abc
from pathlib import Path

import torch as th

from .base import InferenceDataset
from .constants import SpecialToken as ST


class HospitalMortalityBase(InferenceDataset, abc.ABC):
    def __init__(
        self,
        input_dir: str | Path,
        adm_stoken: str,
        dc_stoken: str,
        adm_offset: int = 0,
        n_positions: int = 2048,
        **kwargs,
    ):
        super().__init__(input_dir, n_positions, **kwargs)
        self.stop_stokens = [dc_stoken] + self.stop_stokens

        self.start_indices = self._get_indices_of_stokens(adm_stoken) + adm_offset
        dc_or_dth_indices = self._get_indices_of_stokens([dc_stoken, ST.DEATH])
        self.outcome_indices = self._match(dc_or_dth_indices, self.start_indices)

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, idx) -> tuple[th.Tensor, dict]:
        start_idx = self.start_indices[idx]
        outcome_idx = self.outcome_indices[idx]
        y = {
            #! shift by 2
            "expected": self.vocab.decode(self.tokens[outcome_idx] + 2),
            "true_token_dist": (outcome_idx - start_idx).item(),
            "true_token_time": (self.times[outcome_idx] - self.times[start_idx]).item(),
            "patient_id": self.patient_id_at_idx[start_idx].item(),
            "prediction_time": self.times[start_idx].item(),
            "data_idx": start_idx.item(),
        }
        if self.is_mimic:
            y["hadm_id"] = self._get_hadm_id(start_idx)
            y["icu_stay_id"] = self._get_icu_stay_id(start_idx)

        return super().__getitem__(start_idx), y


class HospitalMortalityDataset(HospitalMortalityBase):
    """Produces timelines that end on inpatient_admission_start token and go back in patients'
    history.

    The target is the inpatient_admission_end (discharge) or death token.
    """

    def __init__(self, input_dir: str | Path, n_positions: int = 2048, **kwargs):
        super().__init__(
            input_dir=input_dir,
            n_positions=n_positions,
            adm_stoken=ST.ADMISSION,
            adm_offset=2,
            dc_stoken=ST.DISCHARGE,
            **kwargs,
        )
