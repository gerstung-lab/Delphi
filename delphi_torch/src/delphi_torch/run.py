from __future__ import annotations

import argparse
from pathlib import Path

from delphi_torch.config import load_config
from delphi_torch.training.loop import fit


def main() -> None:
    parser = argparse.ArgumentParser(description="Delphi Torch training")
    parser.add_argument("--config", type=Path, default=None, help="Path to .toml or .json config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    fit(cfg)


if __name__ == "__main__":
    main()
