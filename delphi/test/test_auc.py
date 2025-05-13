import os

import numpy as np

from archive.evaluate_auc import get_calibration_auc
from delphi import DAYS_PER_YEAR
from delphi.data.dataset import tricolumnar_to_2d
from delphi.data.trajectory import DiseaseRateTrajectory, corrective_indices
from delphi.eval.auc import AgeGroups, CalibrateAUCArgs, auc_by_age_group
from delphi.model.transformer import load_model
from delphi.tokenizer import load_tokenizer_from_ckpt

ckpt = "checkpoints/delphi"
task_input = "forward"
disease = "a41_(other_septicaemia)"
disease = "g30_(alzheimer's_disease)"
disease_lst = "config/disease_list/doi.yaml"

offset = 0.1

auc_args = CalibrateAUCArgs(
    disease_lst=disease_lst,
    age_groups=AgeGroups(start=45, end=85, step=5),
    min_time_gap=offset,
    box_plot=False,
)

model, _ = load_model(ckpt)
tokenizer = load_tokenizer_from_ckpt(ckpt)

logits_path = os.path.join(ckpt, task_input, "logits.bin")
assert os.path.exists(logits_path)
"logits.bin not found in the checkpoint directory"
xt_path = os.path.join(ckpt, task_input, "gen.bin")
assert os.path.exists(xt_path)
"gen.bin not found in the checkpoint directory"

logits = np.fromfile(logits_path, dtype=np.float16).reshape(
    -1, tokenizer.vocab_size + 1
)
XT = np.fromfile(xt_path, dtype=np.uint32).reshape(-1, 3)

X, T = tricolumnar_to_2d(XT)
X_t0, X_t1 = X[:, :-1], X[:, 1:]
T_t0, T_t1 = T[:, :-1], T[:, 1:]


dis_token = tokenizer[disease]

Y = np.zeros_like(X, dtype=np.float16)
sub_idx, pos_idx = np.nonzero(X)
Y[sub_idx, pos_idx] = logits[:, dis_token]
Y = np.exp(Y) * DAYS_PER_YEAR
Y = 1 - np.exp(-Y)

Y_t1 = Y[:, :-1]
C = corrective_indices(T0=T_t0, T1=T_t1, offset=offset)
traj = DiseaseRateTrajectory(
    X_t0=X_t0,
    T_t0=np.take_along_axis(T_t0, C, axis=1),
    X_t1=X_t1,
    T_t1=T_t1,
    Y_t1=Y_t1,
)

gt_age_groups, gt_ctl_counts, gt_dis_counts = get_calibration_auc(
    j=0, k=dis_token, d=(X_t0, T_t0, X_t1, T_t1), p=Y_t1[..., None], offset=offset
)  # type: ignore

age_groups, _, ctl_counts, dis_counts = auc_by_age_group(
    disease_token=dis_token,
    trajectory=traj,
    task_args=auc_args,
)

print("debug")
