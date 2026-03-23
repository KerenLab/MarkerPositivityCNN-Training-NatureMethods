"""
Configure and run marker-positivity training (ResNet + segmentation channels).

``path_to_mantis_dirs`` entries are container or project roots; FOV folders are discovered
(``…/MantisProject…/<FOV>/`` or flat ``…/<FOV>/``). Label CSVs use ``fov``, ``cellID``, ``Marker``, …
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NotRequired, Required, TypedDict

import numpy as np

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
_SAMPLE_DATA_DIR: Final[Path] = _REPO_ROOT / "sample_data"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_tools.dataset_utils import list_fov_dirs_with_segmentation
from training.train_model import train_using_seg

DEFAULT_TRAIN_FOV_FRACTION: Final[float] = 0.9
DEFAULT_NUM_EPOCHS: Final[int] = 10
DEFAULT_BATCH_SIZE: Final[int] = 2048
DEFAULT_NUM_WORKERS: Final[int] = 2
DEFAULT_LEARNING_RATE: Final[float] = 0.0004
DEFAULT_SCHEDULER_GAMMA: Final[float] = 0.9
DEFAULT_DROPOUT: Final[float] = 0.25
DEFAULT_RESNET_SCALE: Final[int] = 18


def resolve_under_sample_data(path: str) -> str:
    if os.path.isabs(path):
        return path
    return str(_SAMPLE_DATA_DIR / path)


def resolve_under_repo(path: str) -> str:
    """Relative paths for outputs and other repo-root files (not under ``sample_data``)."""
    if os.path.isabs(path):
        return path
    return str(_REPO_ROOT / path)


def _fov_train_indices_for_mantis_dirs(
    mantis_dirs: list[str], train_fov_fraction: float
) -> list[np.ndarray]:
    """First ``floor(fraction * n_fovs)`` FOV indices per project (same rule as ``run_train.py``)."""
    out: list[np.ndarray] = []
    for mantis_dir in mantis_dirs:
        fov_dirs = list_fov_dirs_with_segmentation(mantis_dir)
        n_fovs = len(fov_dirs)
        if n_fovs == 0:
            raise ValueError(f"No FOV directories with segmentation under {mantis_dir!r}")
        n_train = max(1, min(int(train_fov_fraction * n_fovs), n_fovs - 1))
        out.append(np.arange(n_train))
    return out


class TrainingConfigDict(TypedDict, total=False):
    path_to_mantis_dirs: Required[list[str]]
    label_csv_path_list: Required[list[str]]
    path_to_output_dir: Required[str]
    model_name: Required[str]
    train_fov_fraction: NotRequired[float]
    pretrained_weights: NotRequired[str | None]
    num_epochs: NotRequired[int]
    batch_size: NotRequired[int]
    num_workers: NotRequired[int]
    learning_rate: NotRequired[float]
    scheduler_gamma: NotRequired[float]
    dropout_rate: NotRequired[float]
    resnet_scale: NotRequired[int]


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    path_to_mantis_dirs: tuple[str, ...]
    label_csv_path_list: tuple[str, ...]
    path_to_output_dir: str
    model_name: str
    train_fov_fraction: float = DEFAULT_TRAIN_FOV_FRACTION
    pretrained_weights: str | None = None
    num_epochs: int = DEFAULT_NUM_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = DEFAULT_NUM_WORKERS
    learning_rate: float = DEFAULT_LEARNING_RATE
    scheduler_gamma: float = DEFAULT_SCHEDULER_GAMMA
    dropout_rate: float = DEFAULT_DROPOUT
    resnet_scale: int = DEFAULT_RESNET_SCALE

    @classmethod
    def from_mapping(cls, m: TrainingConfigDict) -> TrainingConfig:
        try:
            md = m["path_to_mantis_dirs"]
            lc = m["label_csv_path_list"]
            od = m["path_to_output_dir"]
            name = m["model_name"]
        except KeyError as e:
            raise KeyError(f"CONFIG missing required key: {e.args[0]!r}") from e
        return cls(
            path_to_mantis_dirs=tuple(md),
            label_csv_path_list=tuple(lc),
            path_to_output_dir=str(od),
            model_name=str(name),
            train_fov_fraction=float(m.get("train_fov_fraction", DEFAULT_TRAIN_FOV_FRACTION)),
            pretrained_weights=m.get("pretrained_weights"),
            num_epochs=int(m.get("num_epochs", DEFAULT_NUM_EPOCHS)),
            batch_size=int(m.get("batch_size", DEFAULT_BATCH_SIZE)),
            num_workers=int(m.get("num_workers", DEFAULT_NUM_WORKERS)),
            learning_rate=float(m.get("learning_rate", DEFAULT_LEARNING_RATE)),
            scheduler_gamma=float(m.get("scheduler_gamma", DEFAULT_SCHEDULER_GAMMA)),
            dropout_rate=float(m.get("dropout_rate", DEFAULT_DROPOUT)),
            resnet_scale=int(m.get("resnet_scale", DEFAULT_RESNET_SCALE)),
        )


def run_training(config: TrainingConfig) -> None:
    mantis_resolved = [resolve_under_sample_data(p) for p in config.path_to_mantis_dirs]
    labels_resolved = [resolve_under_sample_data(p) for p in config.label_csv_path_list]
    if len(labels_resolved) == 1 and len(mantis_resolved) > 1:
        labels_resolved = labels_resolved * len(mantis_resolved)
    elif len(labels_resolved) != len(mantis_resolved):
        raise ValueError(
            f"Need one label CSV or one per mantis dir; got {len(labels_resolved)} CSV(s) "
            f"for {len(mantis_resolved)} mantis dir(s)."
        )
    out_dir = resolve_under_repo(config.path_to_output_dir)
    os.makedirs(out_dir, exist_ok=True)

    pretrained = (
        resolve_under_sample_data(config.pretrained_weights)
        if config.pretrained_weights
        else None
    )

    print(
        f"Training {config.model_name} → {os.path.abspath(out_dir)} | "
        f"ep={config.num_epochs} batch={config.batch_size} lr={config.learning_rate} "
        f"workers={config.num_workers} dropout={config.dropout_rate} resnet={config.resnet_scale} "
        f"γ={config.scheduler_gamma} fov={config.train_fov_fraction} "
        f"pretrained={pretrained or '—'}"
    )
    print(
        f"  {', '.join(os.path.basename(os.path.normpath(p)) for p in mantis_resolved)} | "
        f"{', '.join(os.path.basename(p) for p in labels_resolved)}"
    )

    fov_inds = _fov_train_indices_for_mantis_dirs(
        mantis_resolved, config.train_fov_fraction
    )

    train_using_seg(
        path_to_mantis_dirs=mantis_resolved,
        label_csv_path_list=labels_resolved,
        fov_inds_for_training_per_project=fov_inds,
        marker_expression_to_filter=None,
        model_w_path_to_load=pretrained,
        path_to_output_dir=out_dir,
        out_nm=config.model_name,
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        learning_rate=config.learning_rate,
        scheduler_gamma=config.scheduler_gamma,
        dropout_rate=config.dropout_rate,
        resnet_scale=config.resnet_scale,
    )


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_NUM_EPOCHS",
    "DEFAULT_TRAIN_FOV_FRACTION",
    "TrainingConfig",
    "TrainingConfigDict",
    "resolve_under_repo",
    "resolve_under_sample_data",
    "run_training",
]
