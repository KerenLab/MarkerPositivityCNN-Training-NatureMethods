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

| File | Use case |
|------|----------|
| `requirements.txt` | Default install (unpinned direct dependencies). |
| `requirements-lock.txt` | Shipped when present: exact `pip freeze` from a tested environment. |

## Run

1. Activate the venv.
2. Place mantis image folders and label CSVs under `sample_data/` when using short paths in CONFIG, or use absolute paths.
3. From the project root, open `train.ipynb`, set **CONFIG**, run all cells.

If you update code or data paths, **restart the Jupyter kernel** before re-running.

## Data format and organization

### Directory structure (images)

- Each **`path_to_mantis_dirs`** entry points at a **container** directory. The container’s **name** must match the **CSV filename prefix** (first two underscore-separated parts), e.g. `Melanoma_Sept1022` for `Melanoma_Sept1022_markerPositiveTrainingTable.csv`.
- **Layout:** `<Container>/MantisProject<Container>/<FOV>/` — the project directory is named `MantisProject` plus the container name (e.g. `MantisProjectMelanoma_Sept1022`). Each **FOV** is a direct child of that project directory (e.g. `Slide01_Point001`). FOV directory names must match the **`fov`** column in the CSV.

**Inside each FOV directory:**

- **Channel (marker) images:** one `.tif` or `.tiff` per channel. The filename stem must match the **`Marker`** column (e.g. `CD3.tif`, `Ki67.tiff`).
- **Segmentation:** **`segmentation_labels.tiff`** (or `.tif`) in the same directory — single-channel image whose pixel values are **cell IDs**.

**Naming summary:** `container_name` = CSV prefix (e.g. `Melanoma_Sept1022`). Project subfolder = `MantisProject` + container name (e.g. `MantisProjectMelanoma_Sept1022`). Internal label keys are `container_name_fov` (e.g. `Melanoma_Sept1022_Slide01_Point001`).

Example layout (paths under `sample_data/` if you use short paths in CONFIG):

```
sample_data/
  Melanoma_Sept1022/                      # container; name = CSV prefix
    MantisProjectMelanoma_Sept1022/
      Slide01_Point001/                   # fov column = "Slide01_Point001"
        CD3.tif
        Ki67.tif
        segmentation_labels.tiff
      Slide01_Point002/
        ...
  Melanoma_Sept1022_markerPositiveTrainingTable.csv
```

### Label CSV

- **Required columns:**
  - **`fov`** — FOV subdirectory name; must match the folder name under the project directory.
  - **`cellID`** — integer; must match a value in `segmentation_labels.tiff`.
  - **`Marker`** — channel name; must match the `.tif` / `.tiff` stem (e.g. `CD3`, `Ki67`).
  - **`Positive`** — `1` = positive; `0` or `-1` = negative (treated as negative in training).
  - **`Manual`** — `1` if manually curated, `0` otherwise.
  - **`InTraining`** — `1` if the row was in baseline training, `0` otherwise (used for weighting and splits).

- **CSV filename and container:** The **container** folder name must match that filename prefix. Example: `Melanoma_Sept1022_…csv` ↔ folder `Melanoma_Sept1022`.

- **Multiple projects:** Provide **one CSV per** `path_to_mantis_dirs` entry, **or** a single CSV reused for all entries (keep CSV order aligned with `path_to_mantis_dirs`).

Example rows:

```text
fov,cellID,Marker,Positive,Manual,InTraining
Slide01_Point002,169,FAP,1,0,0
Slide01_Point002,333,FAP,1,0,0
```

### Model input (for reference)

The pipeline builds 128×128 crops per cell with 3 channels: (1) marker intensity, (2) cell mask, (3) neighbour mask.

## CONFIG (`train.ipynb`)

| Key | Role |
|-----|------|
| `path_to_mantis_dirs` | List of container or project dirs (see layout above). |
| `label_csv_path_list` | One CSV per mantis dir, or one CSV reused for all. |
| `path_to_output_dir` | Where to save `.pth` / `.pt` (relative → project root). |
| `model_name` | Base name; checkpoints are `<model_name>_<timestamp>_epoch_*.pth`. |
| `train_fov_fraction` | Fraction of FOVs used for training (rest for validation). |

Optional: `pretrained_weights`, `num_epochs`, `batch_size`, `num_workers`, `learning_rate`, `scheduler_gamma`, `dropout_rate`, `resnet_scale`.
