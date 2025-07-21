import os
from pathlib import Path

import numpy as np
import pytest
from pytest import Config, FixtureRequest, Metafunc, Parser

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
