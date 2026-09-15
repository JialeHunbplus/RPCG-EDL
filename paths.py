"""Paths for the separated source, dataset, and local weight folders."""

import os
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(
    os.environ.get("RPCG_DATA_ROOT", CODE_ROOT.parent / "2" / "datasets")
).expanduser().resolve()
MOLT5_PATH = Path(
    os.environ.get("MOLT5_PATH", CODE_ROOT.parent / "3" / "molt5-base")
).expanduser().resolve()
ESM2_PATH = Path(
    os.environ.get("ESM2_PATH", CODE_ROOT.parent / "3" / "esm2_weights")
).expanduser().resolve()
