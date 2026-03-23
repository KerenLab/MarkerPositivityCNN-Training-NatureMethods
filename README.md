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
2. Place mantis image folders and label CSVs under `sample_data/` when using short paths in CONFIG, or use absolute paths.
3. From this repository root, open `train.ipynb`, set **CONFIG**, run all cells.

**Rebuild this bundle** from the main `MarkerPositivityCNN` repo after pulling changes:

`cd /path/to/MarkerPositivityCNN && python scripts/build_training_bundle.py`

### Data layout

Each `path_to_mantis_dirs` entry is usually a **container** folder (e.g. `Melanoma_Sept1022`) with:

`…/<Container>/MantisProject<Container>/<FOV_name>/segmentation_labels.tiff` (+ channel `.tif`).

You can also point at the inner `MantisProject…` folder, or use a flat `…/<FOV>/` layout. Training discovers FOV folders the same way as the inference bundle.

### CONFIG (short)


| Key                   | Role                                                               |
| --------------------- | ------------------------------------------------------------------ |
| `path_to_mantis_dirs` | List of container or project dirs (see layout above).              |
| `label_csv_path_list` | One CSV per mantis dir, or one CSV reused for all.                 |
| `path_to_output_dir`  | Where to save `.pth` / `.pt` (relative → repository root).         |
| `model_name`          | Base name; checkpoints are `<model_name>_<timestamp>_epoch_*.pth`. |
| `train_fov_fraction`  | Fraction of FOVs used for training (rest for validation).          |


Optional: `pretrained_weights`, `num_epochs`, `batch_size`, `num_workers`, `learning_rate`, `scheduler_gamma`, `dropout_rate`, `resnet_scale`.