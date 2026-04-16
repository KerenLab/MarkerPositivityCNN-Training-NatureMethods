"""
Configure and run marker-positivity training (ResNet + segmentation channels).

Data input is provided via a training data sources CSV with columns:
``proj_nm``, ``training_labels_path``, ``images_dir``.
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
from scripts.data_validator import validate_training_data_sources
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


def _fov_train_indices_for_image_roots(
    image_roots: list[str], train_fov_fraction: float
) -> list[np.ndarray]:
    """First ``floor(fraction * n_fovs)`` FOV indices per project (same rule as ``run_train.py``)."""
    out: list[np.ndarray] = []
    for root in image_roots:
        fov_dirs = list_fov_dirs_with_segmentation(root)
        n_fovs = len(fov_dirs)
        if n_fovs == 0:
            raise ValueError(f"No FOV directories with segmentation under {root!r}")
        n_train = max(1, min(int(train_fov_fraction * n_fovs), n_fovs - 1))
        out.append(np.arange(n_train))
    return out


class TrainingConfigDict(TypedDict, total=False):
    training_data_sources_csv: Required[str]
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
    training_data_sources_csv: str
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
            training_data_sources_csv = m["training_data_sources_csv"]
            od = m["path_to_output_dir"]
            name = m["model_name"]
        except KeyError as e:
            raise KeyError(f"CONFIG missing required key: {e.args[0]!r}") from e
        return cls(
            training_data_sources_csv=str(training_data_sources_csv),
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
    data_sources_resolved = resolve_under_repo(config.training_data_sources_csv)
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
    print(f"  training_data_sources_csv={data_sources_resolved}")

    validation = validate_training_data_sources(data_sources_resolved)
    roots_resolved = validation.images_dir_list
    labels_resolved = validation.training_labels_path_list
    validation_summary = validation.summary
    print(
        "Data validation passed "
        f"(training_data_sources_csvs={validation_summary.training_data_sources_csvs_checked}, "
        f"csvs={validation_summary.csv_files_checked}, "
        f"images_dirs={validation_summary.images_dirs_checked}, "
        f"fovs={validation_summary.fov_count}, rows={validation_summary.rows_checked})."
    )

    fov_inds = _fov_train_indices_for_image_roots(
        roots_resolved, config.train_fov_fraction
    )

    train_using_seg(
        images_root_folder=roots_resolved,
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
    "run_training",
]
