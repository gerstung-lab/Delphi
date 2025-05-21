import scipy.stats
import scipy
import warnings
import torch
from model import DelphiConfig, Delphi
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from utils import get_batch, get_p2i
from pathlib import Path


def auc(x1, x2):
    n1 = len(x1)
    n2 = len(x2)
    R1 = np.concatenate([x1, x2]).argsort().argsort()[:n1].sum() + n1
    U1 = n1 * n2 + 0.5 * n1 * (n1 + 1) - R1
    if n1 == 0 or n2 == 0:
        return np.nan
    return U1 / n1 / n2


def get_common_diseases(delphi_labels, filter_min_total=100):
    chapters_of_interest = [
        "I. Infectious Diseases",
        "II. Neoplasms",
        "III. Blood & Immune Disorders",
        "IV. Metabolic Diseases",
        "V. Mental Disorders",
        "VI. Nervous System Diseases",
        "VII. Eye Diseases",
        "VIII. Ear Diseases",
        "IX. Circulatory Diseases",
        "X. Respiratory Diseases",
        "XI. Digestive Diseases",
        "XII. Skin Diseases",
        "XIII. Musculoskeletal Diseases",
        "XIV. Genitourinary Diseases",
        "XV. Pregnancy & Childbirth",
        "XVI. Perinatal Conditions",
        "XVII. Congenital Abnormalities",
        "Death",
    ]
    labels_df = delphi_labels[
        delphi_labels["ICD-10 Chapter (short)"].isin(chapters_of_interest) * (delphi_labels["count"] > filter_min_total)
    ]
    return labels_df["index"].tolist()


def optimized_bootstrapped_auc_gpu(case, control, n_bootstrap=1000):
    """
    Computes bootstrapped AUC estimates using PyTorch on CUDA.

    Parameters:
        case: 1D tensor of scores for positive cases
        control: 1D tensor of scores for controls
        n_bootstrap: Number of bootstrap replicates

    Returns:
        Tensor of shape (n_bootstrap,) containing AUC for each bootstrap replicate
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This function requires a GPU.")

    # Convert inputs to CUDA tensors
    if not torch.is_tensor(case):
        case = torch.tensor(case, device="cuda", dtype=torch.float32)
    else:
        case = case.to("cuda", dtype=torch.float32)

    if not torch.is_tensor(control):
        control = torch.tensor(control, device="cuda", dtype=torch.float32)
    else:
        control = control.to("cuda", dtype=torch.float32)

    n_case = case.size(0)
    n_control = control.size(0)
    total = n_case + n_control

    # Generate bootstrap samples
    boot_idx_case = torch.randint(0, n_case, (n_bootstrap, n_case), device="cuda")
    boot_idx_control = torch.randint(0, n_control, (n_bootstrap, n_control), device="cuda")

    boot_case = case[boot_idx_case]
    boot_control = control[boot_idx_control]

    combined = torch.cat([boot_case, boot_control], dim=1)

    # Mask to identify case entries
    mask = torch.zeros((n_bootstrap, total), dtype=torch.bool, device="cuda")
    mask[:, :n_case] = True

    # Compute ranks and AUC
    ranks = combined.argsort(dim=1).argsort(dim=1)
    case_ranks_sum = torch.sum(ranks.float() * mask.float(), dim=1)
    min_case_rank_sum = n_case * (n_case - 1) / 2.0
    U = case_ranks_sum - min_case_rank_sum
    aucs = U / (n_case * n_control)
    return aucs.cpu().tolist()


# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float32)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float32)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float32)
    ty = np.empty([k, n], dtype=np.float32)
    tz = np.empty([k, m + n], dtype=np.float32)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def get_auc_delong_var(healthy_scores, diseased_scores):
    """
    Computes ROC AUC value and variance using DeLong's method

    Args:
        healthy_scores: Values for class 0 (healthy/controls)
        diseased_scores: Values for class 1 (diseased/cases)
    Returns:
        AUC value and variance
    """
    # Create ground truth labels (1 for diseased, 0 for healthy)
    ground_truth = np.array([1] * len(diseased_scores) + [0] * len(healthy_scores))
    predictions = np.concatenate([diseased_scores, healthy_scores])

    # Compute statistics needed for DeLong method
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]

    # Calculate AUC and covariance
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"

    return aucs[0], delongcov


def get_calibration_auc(j, k, d, p, offset=365.25, age_groups=range(45, 80, 5), precomputed_idx=None, n_bootstrap=1, use_delong=False):
    age_step = age_groups[1] - age_groups[0]

    # Indexes of cases with disease k
    wk = np.where(d[2] == k)

    if len(wk[0]) < 2:
        return None

    # For controls, we need to exclude cases with disease k
    wc = np.where((d[2] != k) * (~(d[2] == k).any(-1))[..., None])

    wall = (np.concatenate([wk[0], wc[0]]), np.concatenate([wk[1], wc[1]]))  # All cases and controls

    # We need to take into account the offset t and use the tokens for prediction that are at least t before the event
    if precomputed_idx is None:
        pred_idx = (d[1][wall[0]] <= d[3][wall].reshape(-1, 1) - offset).sum(1) - 1
    else:
        pred_idx = precomputed_idx[wall]  # It's actually much faster to precompute this

    z = d[1][(wall[0], pred_idx)]  # Times of the tokens for prediction
    z = z[pred_idx != -1]

    zk = d[3][wall]  # Target times
    zk = zk[pred_idx != -1]

    # x = np.exp(p[..., j][(wall[0], pred_idx)]) * 365.25
    # x = 1 - np.exp(-x * age_step)  # the function is monotinic, so we don't need to do this for the AUC
    x = p[..., j][(wall[0], pred_idx)]
    x = x[pred_idx != -1]

    wk = (wk[0][pred_idx[: len(wk[0])] != -1], wk[1][pred_idx[: len(wk[0])] != -1])
    p_idx = wall[0][pred_idx != -1]

    out = []

    for i, aa in enumerate(age_groups):
        a = np.logical_and(z / 365.25 >= aa, z / 365.25 < aa + age_step)
        # Optionally, add extra filtering on the time difference, for example:
        # a *= (zk - z < 365.25)
        selected_groups = p_idx[a]
        perm = np.random.permutation(len(selected_groups))
        _, indices = np.unique(selected_groups[perm], return_index=True)
        indices = perm[indices]
        selected = np.zeros(np.sum(a), dtype=bool)
        selected[indices] = True
        a[a] = selected

        control = x[len(wk[0]) :][a[len(wk[0]) :]]
        case = x[: len(wk[0])][a[: len(wk[0])]]

        if len(control) == 0 or len(case) == 0:
            continue

        if use_delong:
            auc_value_delong, auc_variance_delong = get_auc_delong_var(control, case)
            auc_delong_dict = {"auc_delong": auc_value_delong, "auc_variance_delong": auc_variance_delong}
        else:
            auc_delong_dict = {}

        if n_bootstrap > 1:
            aucs_bootstrapped = optimized_bootstrapped_auc_gpu(case, control, n_bootstrap)

        for bootstrap_idx in range(n_bootstrap):
            y = auc_value_delong if n_bootstrap == 1 else aucs_bootstrapped[bootstrap_idx]
            out_item = {
                "token": k,
                "auc": y,
                "age": aa,
                "n_healthy": len(control),
                "n_diseased": len(case),
            }
            out.append(out_item | auc_delong_dict)
            if n_bootstrap > 1:
                out_item["bootstrap_idx"] = bootstrap_idx
    return out


# New internal function that performs the AUC evaluation pipeline.
def evaluate_auc_pipeline(
    model,
    d100k,
    output_path,
    delphi_labels,
    diseases_of_interest=None,
    filter_min_total=100,
    disease_chunk_size=200,
    age_groups=np.arange(40, 80, 5),
    offset=0.1,
    batch_size=128,
    device="cpu",
    seed=1337,
    n_bootstrap=1,
    meta_info={},
):
    """
    Runs the AUC evaluation pipeline.

    Args:
        model (torch.nn.Module): The loaded model set to eval().
        d100k (tuple): Data batch from get_batch.
        delphi_labels (pd.DataFrame): DataFrame with label info (token names, etc. "delphi_labels_chapters_colours_icd.csv").
        output_path (str | None): Directory where CSV files will be written. If None, files will not be saved.
        diseases_of_interest (np.ndarray or list, optional): If provided, these disease indices are used.
        filter_min_total (int): Minimum total token count to include a token.
        disease_chunk_size (int): Maximum chunk size for processing diseases.
        age_groups (np.ndarray): Age groups to use in calibration.
        offset (float): Offset used in get_calibration_auc.
        batch_size (int): Batch size for model forwarding.
        device (str): Device identifier.
        seed (int): Random seed for reproducibility.
        n_bootstrap (int): Number of bootstrap samples. (1 for no bootstrap)
    Returns:
        tuple: (df_auc_unpooled, df_auc, df_both) DataFrames.
    """

    assert n_bootstrap > 0, "n_bootstrap must be greater than 0"

    # Set random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Get common diseases
    if diseases_of_interest is None:
        diseases_of_interest = get_common_diseases(delphi_labels, filter_min_total)

    # Split diseases into chunks for processing
    num_chunks = (len(diseases_of_interest) + disease_chunk_size - 1) // disease_chunk_size
    diseases_chunks = np.array_split(diseases_of_interest, num_chunks)

    # Precompute prediction indices for calibration
    pred_idx_precompute = (d100k[1][:, :, np.newaxis] < d100k[3][:, np.newaxis, :] - offset).sum(1) - 1

    all_aucs = []
    tqdm_options = {"desc": "Processing disease chunks", "total": len(diseases_chunks)}
    for disease_chunk_idx, diseases_chunk in tqdm(enumerate(diseases_chunks), **tqdm_options):
        p100k = []
        model.to(device)
        with torch.no_grad():
            # Process the evaluation data in batches
            for dd in tqdm(
                zip(*[torch.split(x, batch_size) for x in d100k]),
                desc=f"Model inference, chunk {disease_chunk_idx}",
                total=d100k[0].shape[0] // batch_size + 1,
            ):
                dd = [x.to(device) for x in dd]
                outputs = model(*dd)[0].cpu().detach().numpy()
                # Keep only the columns corresponding to the current disease chunk
                p100k.append(outputs[:, :, diseases_chunk].astype("float16"))  # enough to store logits, but not rates
        p100k = np.vstack(p100k)

        # Loop over each disease (token) in the current chunk, sexes separately
        for sex, sex_idx in [("female", 2), ("male", 3)]:
            sex_mask = ((d100k[0] == sex_idx).sum(1) > 0).cpu().detach().numpy()
            p_sex = p100k[sex_mask]
            d100k_sex = [d_[sex_mask].cpu().detach().numpy() for d_ in d100k]
            precomputed_idx_subset = pred_idx_precompute[sex_mask].cpu().detach().numpy()
            for j, k in tqdm(
                list(enumerate(diseases_chunk)), desc=f"Processing diseases in chunk {disease_chunk_idx}, {sex}"
            ):
                # Get calibration AUC for the current disease token.
                out = get_calibration_auc(
                    j,
                    k,
                    d100k_sex,
                    p_sex,
                    age_groups=age_groups,
                    offset=offset,
                    precomputed_idx=precomputed_idx_subset,
                    n_bootstrap=n_bootstrap,
                    use_delong=True,
                )
                if out is None:
                    # print(f"No data for disease {k} and sex {sex}")
                    continue
                for out_item in out:
                    out_item["sex"] = sex
                    all_aucs.append(out_item)

    df_auc_unpooled = pd.DataFrame(all_aucs)

    for key, value in meta_info.items():
        df_auc_unpooled[key] = value

    delphi_labels_subset = delphi_labels[['index', 'ICD-10 Chapter (short)', 'name', 'color', 'count']]
    df_auc_unpooled_merged = df_auc_unpooled.merge(delphi_labels_subset, left_on="token", right_on="index", how="inner")

    def aggregate_age_brackets_delong(group):
        # For normal distributions, when averaging n of them:
        # The variance of the sum is the sum of variances
        # The variance of the average is the sum of variances divided by n^2
        n = len(group)
        mean = group['auc_delong'].mean()
        # Since we're taking the average, divide combined variance by n^2
        var = group['auc_variance_delong'].sum() / (n**2)
        return pd.Series({
            'auc': mean,
            'auc_variance_delong': var,
            'n_samples': n, 
            'n_diseased': group['n_diseased'].sum(),
            'n_healthy': group['n_healthy'].sum(),
        })

    print('Using DeLong method to calculate AUC confidence intervals..')
    
    df_auc = df_auc_unpooled.groupby(["token"]).apply(aggregate_age_brackets_delong).reset_index()
    df_auc_merged = df_auc.merge(delphi_labels, left_on="token", right_on="index", how="inner")
    
    if output_path is not None:
        Path(output_path).mkdir(exist_ok=True, parents=True)
        df_auc_merged.to_parquet(f"{output_path}/df_both.parquet", index=False)
        df_auc_unpooled_merged.to_parquet(f"{output_path}/df_auc_unpooled.parquet", index=False)

    return df_auc_unpooled_merged, df_auc_merged


def main():
    parser = argparse.ArgumentParser(description="Evaluate AUC")
    parser.add_argument("--input_path", type=str, help="Path to the dataset")
    parser.add_argument("--output_path", type=str, help="Path to the output")
    parser.add_argument("--model_ckpt_path", type=str, help="Path to the model weights")
    parser.add_argument("--no_event_token_rate", type=int, help="No event token rate")
    parser.add_argument(
        "--health_token_replacement_prob", default=0.0, type=float, help="Health token replacement probability"
    )
    parser.add_argument("--dataset_subset_size", type=int, default=-1, help="Dataset subset size for evaluation")
    parser.add_argument("--n_bootstrap", type=int, default=1, help="Number of bootstrap samples")
    # Optional filtering/chunking parameters:
    parser.add_argument("--filter_min_total", type=int, default=100, help="Minimum total count to filter tokens")
    parser.add_argument("--disease_chunk_size", type=int, default=200, help="Chunk size for processing diseases")
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    no_event_token_rate = args.no_event_token_rate
    health_token_replacement_prob = args.health_token_replacement_prob
    dataset_subset_size = args.dataset_subset_size

    # Create output folder if it doesn't exist.
    Path(output_path).mkdir(exist_ok=True, parents=True)

    device = "cuda"
    seed = 1337

    # Load model checkpoint and initialize model.
    ckpt_path = args.model_ckpt_path
    checkpoint = torch.load(ckpt_path, map_location=device)
    conf = DelphiConfig(**checkpoint["model_args"])
    model = Delphi(conf)
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)

    # Load training and validation data.
    val = np.fromfile(f"{input_path}/val.bin", dtype=np.uint32).reshape(-1, 3).astype(np.int64)

    val_p2i = get_p2i(val)

    if dataset_subset_size == -1:
        dataset_subset_size = len(val_p2i)

    # Get a subset batch for evaluation.
    d100k = get_batch(
        range(dataset_subset_size),
        val,
        val_p2i,
        select="left",
        block_size=80,
        device=device,
        padding="random",
        no_event_token_rate=no_event_token_rate,
        health_token_replacement_prob=health_token_replacement_prob,
    )

    # Load labels (external) to be passed in.
    delphi_labels = pd.read_csv("delphi_labels_chapters_colours_icd.csv")

    # Call the internal evaluation function.
    df_auc_unpooled, df_auc_merged = evaluate_auc_pipeline(
        model,
        d100k,
        output_path,
        delphi_labels,
        diseases_of_interest=None,
        filter_min_total=args.filter_min_total,
        disease_chunk_size=args.disease_chunk_size,
        device=device,
        seed=seed,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
