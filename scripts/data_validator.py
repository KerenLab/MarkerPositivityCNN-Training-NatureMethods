from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

REQUIRED_LABEL_COLUMNS = ("fov", "cellID", "Marker", "Positive", "Manual", "InTraining")
REQUIRED_MANIFEST_COLUMNS = ("proj_nm", "training_labels_path", "images_dir")


@dataclass(frozen=True, slots=True)
class DatasetValidationSummary:
    training_data_sources_csvs_checked: int
    csv_files_checked: int
    images_dirs_checked: int
    fov_count: int
    rows_checked: int


@dataclass(frozen=True, slots=True)
class ManifestResolution:
    training_labels_path_list: list[str]
    images_dir_list: list[str]
    summary: DatasetValidationSummary


def _fmt_list(values: Iterable[object], limit: int = 8) -> str:
    vals = sorted(set(str(v) for v in values))
    if len(vals) <= limit:
        return ", ".join(vals)
    return ", ".join(vals[:limit]) + f", ... (+{len(vals) - limit} more)"


def _is_tif(name: str) -> bool:
    lower = name.lower()
    return lower.endswith(".tif") or lower.endswith(".tiff")


def _marker_name_from_filename(name: str) -> str:
    low = name.lower()
    if low.endswith(".tiff"):
        return name[:-5]
    if low.endswith(".tif"):
        return name[:-4]
    return name


def _to_abs_path(base_dir: str, path: str) -> str:
    path = str(path).strip()
    if os.path.isabs(path):
        return os.path.abspath(os.path.normpath(path))
    return os.path.abspath(os.path.normpath(os.path.join(base_dir, path)))


def validate_training_data_sources(training_data_sources_csv_path: str) -> ManifestResolution:
    errors: list[str] = []
    warnings: list[str] = []
    total_rows_checked = 0
    total_fovs = 0

    data_sources_abs = os.path.abspath(os.path.normpath(training_data_sources_csv_path))
    if not os.path.isfile(data_sources_abs):
        raise ValueError(f"Training data sources CSV does not exist: {data_sources_abs}")

    try:
        manifest_df = pd.read_csv(data_sources_abs)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Cannot read training data sources CSV {data_sources_abs}: {e}") from e

    if manifest_df.empty:
        raise ValueError(f"Training data sources CSV is empty: {data_sources_abs}")

    missing_manifest_cols = [c for c in REQUIRED_MANIFEST_COLUMNS if c not in manifest_df.columns]
    if missing_manifest_cols:
        raise ValueError(
            f"Training data sources CSV {data_sources_abs} is missing required columns: {', '.join(missing_manifest_cols)} "
            f"(required: {', '.join(REQUIRED_MANIFEST_COLUMNS)})."
        )

    base_dir = os.path.dirname(data_sources_abs)
    labels_list: list[str] = []
    images_list: list[str] = []

    for row_idx, row in manifest_df.iterrows():
        row_num = row_idx + 2  # Header is row 1.
        training_labels_raw = str(row["training_labels_path"]).strip()
        images_dir_raw = str(row["images_dir"]).strip()
        training_labels_path = _to_abs_path(base_dir, training_labels_raw)
        images_dir = _to_abs_path(base_dir, images_dir_raw)
        labels_list.append(training_labels_path)
        images_list.append(images_dir)

        if not os.path.isabs(training_labels_raw):
            errors.append(
                f"Training data sources row {row_num}: training_labels_path must be a full path, got {training_labels_raw!r}"
            )
        if not os.path.isabs(images_dir_raw):
            errors.append(
                f"Training data sources row {row_num}: images_dir must be a full path, got {images_dir_raw!r}"
            )

        if not os.path.isfile(training_labels_path):
            errors.append(
                f"Training data sources row {row_num}: training_labels_path does not exist: {training_labels_path}"
            )
            continue
        if not os.path.isdir(images_dir):
            errors.append(f"Training data sources row {row_num}: images_dir does not exist: {images_dir}")
            continue

        try:
            df = pd.read_csv(training_labels_path)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{training_labels_path}: cannot read CSV ({e})")
            continue

        total_rows_checked += len(df)
        if df.empty:
            errors.append(f"{training_labels_path}: CSV is empty.")
            continue

        missing_cols = [c for c in REQUIRED_LABEL_COLUMNS if c not in df.columns]
        if missing_cols:
            errors.append(
                f"{training_labels_path}: missing required columns: {', '.join(missing_cols)} "
                f"(required: {', '.join(REQUIRED_LABEL_COLUMNS)})."
            )
            continue

        fov_series = df["fov"].astype(str).str.strip()
        marker_series = df["Marker"].astype(str).str.strip()
        if (fov_series == "").any():
            errors.append(f"{training_labels_path}: column 'fov' contains empty values.")
        if (marker_series == "").any():
            errors.append(f"{training_labels_path}: column 'Marker' contains empty values.")

        cell_id_num = pd.to_numeric(df["cellID"], errors="coerce")
        if cell_id_num.isna().any():
            errors.append(f"{training_labels_path}: column 'cellID' contains non-numeric values.")
        elif (cell_id_num < 0).any():
            errors.append(f"{training_labels_path}: column 'cellID' contains negative values.")

        for col in ("Manual", "InTraining"):
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.isna().any():
                errors.append(f"{training_labels_path}: column '{col}' contains non-numeric values.")
                continue
            bad = sorted(set(vals.astype(int).unique()) - {0, 1})
            if bad:
                errors.append(
                    f"{training_labels_path}: column '{col}' must contain only 0/1, found: {_fmt_list(bad)}"
                )

        pos_vals = pd.to_numeric(df["Positive"], errors="coerce")
        if pos_vals.isna().any():
            errors.append(f"{training_labels_path}: column 'Positive' contains non-numeric values.")

        csv_fovs = set(fov_series.unique())
        total_fovs += len(csv_fovs)
        dir_entries = [n for n in os.listdir(images_dir) if not n.startswith(".")]
        dir_paths = [os.path.join(images_dir, n) for n in dir_entries]

        non_dir_entries = [n for n, p in zip(dir_entries, dir_paths) if not os.path.isdir(p)]
        if non_dir_entries:
            errors.append(
                f"{images_dir}: images_dir must contain only FOV directories, found non-directory entries: "
                f"{_fmt_list(non_dir_entries)}"
            )

        image_fovs = {n for n, p in zip(dir_entries, dir_paths) if os.path.isdir(p)}
        missing_in_dir = sorted(csv_fovs - image_fovs)
        if missing_in_dir:
            errors.append(
                f"{training_labels_path}: CSV has FOV(s) missing in images_dir {images_dir}: "
                f"{_fmt_list(missing_in_dir)}"
            )

        csv_norm = df.copy()
        csv_norm["fov"] = fov_series
        csv_norm["Marker"] = marker_series

        for fov_name in sorted(csv_fovs & image_fovs):
            fov_dir = os.path.join(images_dir, fov_name)
            fov_rows = csv_norm[csv_norm["fov"] == fov_name]
            csv_markers = set(fov_rows["Marker"].unique())
            tif_files = [n for n in os.listdir(fov_dir) if _is_tif(n)]
            non_tif_files = [
                n
                for n in os.listdir(fov_dir)
                if not n.startswith(".") and not _is_tif(n)
            ]
            if non_tif_files:
                errors.append(
                    f"{fov_dir}: contains non-tif files. Allowed files are marker .tif/.tiff and "
                    f"segmentation_labels.tif/.tiff only. Found: {_fmt_list(non_tif_files)}"
                )

            seg_files = [n for n in tif_files if _marker_name_from_filename(n) == "segmentation_labels"]
            if len(seg_files) != 1:
                errors.append(
                    f"{fov_dir}: expected exactly one segmentation_labels tif/tiff file, found {len(seg_files)}."
                )

            image_markers = {
                _marker_name_from_filename(n)
                for n in tif_files
                if _marker_name_from_filename(n) != "segmentation_labels"
            }

            missing_marker_images = sorted(csv_markers - image_markers)
            if missing_marker_images:
                errors.append(
                    f"{training_labels_path}: fov '{fov_name}' is missing marker image files in {fov_dir}: "
                    f"{_fmt_list(missing_marker_images)}"
                )

    if errors:
        msg = ["Data validation failed. Fix the issues below before training:\n"]
        msg.extend(f"- {e}" for e in errors)
        if warnings:
            msg.append("\nAdditional warnings:")
            msg.extend(f"- {w}" for w in warnings)
        raise ValueError("\n".join(msg))

    if warnings:
        print("Data validation warnings:")
        for w in warnings:
            print(f"- {w}")

    summary = DatasetValidationSummary(
        training_data_sources_csvs_checked=1,
        csv_files_checked=len(labels_list),
        images_dirs_checked=len(images_list),
        fov_count=total_fovs,
        rows_checked=total_rows_checked,
    )
    return ManifestResolution(
        training_labels_path_list=labels_list,
        images_dir_list=images_list,
        summary=summary,
    )


__all__ = [
    "DatasetValidationSummary",
    "ManifestResolution",
    "validate_training_data_sources",
]
