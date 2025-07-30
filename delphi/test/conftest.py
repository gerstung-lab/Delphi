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


@pytest.fixture(scope="class", params=[1, 10, 50])
def batch_size(request: FixtureRequest):
    return request.param


@pytest.fixture(scope="class", params=[5, 100])
def vocab_size(request: FixtureRequest):
    return request.param


@pytest.fixture(scope="class", params=[10, 100])
def seq_len(request: FixtureRequest):
    return request.param


@pytest.fixture(scope="class", params=[(1, 500), (5, 200), (8, 300)])
def time_bins(request: FixtureRequest):
    n_bins, bin_width = request.param
    return torch.Tensor(np.array([i * bin_width for i in np.arange(n_bins + 1)]))


@pytest.fixture(scope="class", params=[1, 20])
def n_tasks(request: FixtureRequest):
    return request.param


class CaseGenerator:

    def __init__(self, batch_size: int, seq_len: int, vocab_size: int, n_tasks: int):

        max_age = 100 * DAYS_PER_YEAR
        torch.manual_seed(42)

        sample_shape = (batch_size, seq_len)
        timesteps = torch.randint(0, int(max_age), sample_shape)
        self.timesteps, _ = torch.sort(timesteps, dim=1)
        self.tokens = torch.randint(1, vocab_size, sample_shape)

        max_pad = int(seq_len / 2)
        pad_idx = torch.randint(0, max_pad, (batch_size,))
        pad_mask = torch.arange(seq_len).reshape(1, -1) <= pad_idx.reshape(-1, 1)

        self.tokens[pad_mask] = 0
        self.timesteps[pad_mask] = -1e4

        self.vocab_size = vocab_size
        self.n_tasks = min(n_tasks, vocab_size - 1)

    @property
    def targets(self):
        return self.tokens[:, 1:]

    @property
    def age(self):
        return self.timesteps[:, :-1]

    @property
    def targets_age(self):
        return self.timesteps[:, 1:]

    @property
    def task_tokens(self):
        pdf = torch.ones((self.vocab_size))
        pdf[0] = 0
        tokens = torch.multinomial(pdf, num_samples=self.n_tasks, replacement=False)
        return tokens


@pytest.fixture(scope="class")
def case_generator(batch_size: int, seq_len: int, vocab_size: int, n_tasks: int):
    return CaseGenerator(
        batch_size=batch_size, seq_len=seq_len, vocab_size=vocab_size, n_tasks=n_tasks
    )


@pytest.fixture(scope="class")
def targets(case_generator: CaseGenerator):
    return case_generator.targets


@pytest.fixture(scope="class")
def targets_age(case_generator: CaseGenerator):
    return case_generator.targets_age


@pytest.fixture(scope="class")
def age(case_generator: CaseGenerator):
    return case_generator.age


@pytest.fixture(scope="class")
def task_tokens(case_generator: CaseGenerator):
    return case_generator.task_tokens


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
