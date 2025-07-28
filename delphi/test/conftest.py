import os
from pathlib import Path

import numpy as np
import pytest
import torch
from pytest import Config, FixtureRequest, Metafunc, Parser

from delphi import DAYS_PER_YEAR
from delphi.env import DELPHI_DATA_DIR


def pytest_addoption(parser: Parser):

    parser.addoption(
        "--dataset",
        action="store",
        default="ukb_real_data",
        help="directory to data to be tested",
    )


def get_dataset_dir(config: Config):

    dataset = config.getoption("--dataset")

    return Path(DELPHI_DATA_DIR) / str(dataset)


@pytest.fixture
def dataset_dir(request: FixtureRequest):

    return get_dataset_dir(request.config)


@pytest.fixture
def all_participants(dataset_dir):

    participant_list_dir = Path(dataset_dir) / "participants"
    participants = set()
    for participant_list in participant_list_dir.rglob("*.bin"):
        participants_np = np.fromfile(participant_list, dtype=np.uint32)
        participants = participants.union(set(participants_np.tolist()))

    return np.array(list(participants))


def generate_samples():

    torch.manual_seed(42)

    batch_size = np.arange(1, 11)
    vocab_size = np.arange(5, 110, step=10)
    seq_len = np.arange(11, 21)
    max_age = int(100 * DAYS_PER_YEAR)

    age_list = []
    targets_age_list = []
    targets_list = []
    for b, l, v in zip(batch_size, seq_len, vocab_size):
        sample_shape = (b, l)
        timesteps = torch.randint(0, max_age, sample_shape)
        timesteps, _ = torch.sort(timesteps, dim=1)
        tokens = torch.randint(1, v, sample_shape)

        max_pad = int(l / 2)

        pad_idx = torch.randint(0, max_pad, (b,))
        pad_mask = torch.arange(l).reshape(1, -1) <= pad_idx.reshape(-1, 1)

        tokens[pad_mask] = 0
        timesteps[pad_mask] = -1e4

        age, targets_age = timesteps[:, :-1], timesteps[:, 1:]
        _, targets = tokens[:, :-1], tokens[:, 1:]

        age_list.append(age)
        targets_age_list.append(targets_age)
        targets_list.append(targets)

    return age_list, targets_age_list, targets_list, vocab_size.tolist()


def pytest_generate_tests(metafunc: Metafunc):

    if "expansion_pack_path" in metafunc.fixturenames:

        dataset_dir = get_dataset_dir(metafunc.config)
        pack_dir = os.path.join(dataset_dir, "expansion_packs")
        _, expansion_packs, _ = next(os.walk(pack_dir))

        metafunc.parametrize(
            "expansion_pack_path",
            [
                os.path.join(pack_dir, expansion_pack)
                for expansion_pack in expansion_packs
            ],
        )

    if set(metafunc.fixturenames) == set(["age", "targets_age"]):

        age_list, targets_age_list, _, _ = generate_samples()
        test_params = list(zip(age_list, targets_age_list))
        metafunc.parametrize("age,targets_age", test_params)

    if set(metafunc.fixturenames) == set(
        ["age", "targets_age", "targets", "vocab_size"]
    ):

        age_list, targets_age_list, targets_list, vocab_size = generate_samples()
        test_params = list(zip(age_list, targets_age_list, targets_list, vocab_size))
        metafunc.parametrize("age,targets_age,targets,vocab_size", test_params)
