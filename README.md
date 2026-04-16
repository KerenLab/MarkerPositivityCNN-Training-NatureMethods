# Marker positivity — training

Train the ResNet model on labelled multiplex FOV data. Checkpoints (`.pth`) and a TorchScript export (`.pt`) are written to the configured output directory.

## Python version

Use **Python 3.11** (see `.python-version`).

## Installation (venv)

**macOS / Linux**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install jupyter
```

**Windows (PowerShell)**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install jupyter
```


| File                    | Use case                                                            |
| ----------------------- | ------------------------------------------------------------------- |
| `requirements.txt`      | Default install (unpinned direct dependencies).                     |
| `requirements-lock.txt` | Shipped when present: exact `pip freeze` from a tested environment. |


## Run

1. Activate the venv.
2. Create a `training_data_sources_csv` file.
3. From the project root, open `train.ipynb`, set **CONFIG**, run all cells.

If you update code or data paths, **restart the Jupyter kernel** before re-running.

## Data format and organization

### `training_data_sources_csv`

Required columns:

- `proj_nm` - kept for reference only; not validated or used by training.
- `training_labels_path` - absolute path to a label CSV.
- `images_dir` - absolute path to a directory whose immediate subfolders are FOV folders.

Example:

```
proj_nm,training_labels_path,images_dir
Melanoma_Sept1022,/abs/path/labels.csv,/abs/path/MyMelanoma_Sept1022
```

### Label CSV

Required columns:

- `fov` - FOV folder name inside `images_dir`.
- `cellID` - integer cell identifier.
- `Marker` - marker image stem, for example `CD3` for `CD3.tif`.
- `Positive` - numeric label; `1` is positive and other values are treated as negative.
- `Manual` - must be `0` or `1`.
- `InTraining` - must be `0` or `1`.

Example:

```text
fov,cellID,Marker,Positive,Manual,InTraining
Slide01_Point002,169,FAP,1,0,0
Slide01_Point002,333,FAP,1,0,0
```

### Images directory

- `images_dir` must contain only FOV subdirectories.
- Every CSV `fov` must exist as a folder in `images_dir`.
- Inside each FOV folder:
  - marker images are `.tif` or `.tiff`
  - `segmentation_labels.tif` or `segmentation_labels.tiff` must exist exactly once
  - every CSV `Marker` used for that FOV must have a matching image file

Example:

```text
/abs/path/MyMelanoma_Sept1022/
  Slide01_Point001/
    FAP.tif
    CD20.tif
    segmentation_labels.tiff
  Slide01_Point002/
    CD20.tif
    CD8.tif
    segmentation_labels.tif
```

### Model input (for reference)

The pipeline builds 128×128 crops per cell with 3 channels: (1) marker intensity, (2) cell mask, (3) neighbour mask.

## CONFIG (`train.ipynb`)


| Key                   | Role                                                               |
| --------------------- | ------------------------------------------------------------------ |
| `training_data_sources_csv` | Path to the training data sources CSV.                      |
| `path_to_output_dir`  | Where to save `.pth` / `.pt` (relative → project root).            |
| `model_name`          | Base name; checkpoints are `<model_name>_<timestamp>_epoch_*.pth`. |
| `train_fov_fraction`  | Fraction of FOVs used for training (rest for validation).          |


Optional: `pretrained_weights`, `num_epochs`, `batch_size`, `num_workers`, `learning_rate`, `scheduler_gamma`, `dropout_rate`, `resnet_scale`.
