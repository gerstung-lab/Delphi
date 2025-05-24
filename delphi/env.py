import os

try:
    DELPHI_DATA_DIR = os.environ["DELPHI_DATA_DIR"]
    DELPHI_CKPT_DIR = os.environ["DELPHI_CKPT_DIR"]
except KeyError as e:
    raise EnvironmentError(f"required environment variable(s) not set: {e}")
