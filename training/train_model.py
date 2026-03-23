import os
import time
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler

from data_tools.channels_data import MarkerType
from data_tools.datasets import FOVDatasetByLabelDict, create_train_indices, parse_label_csv_files
from data_tools.dataset_utils import (
    get_mantis_dir_path_from_proj_name,
    get_path_to_label_csv,
    list_fov_dirs_with_segmentation,
)
from data_tools.torch_models import ResNet18CellWithSegFeatureExtractor
from evaluation.eval_model import eval_model
from evaluation.perf_calculator import PerfCalculator


def train_using_seg(
    path_to_mantis_dirs: List[str],
    label_csv_path_list: List[str],
    fov_inds_for_training_per_project: List[np.ndarray],
    marker_expression_to_filter: Optional[List[MarkerType]] = None,
    model_w_path_to_load: Optional[str] = None,
    path_to_output_dir: Optional[str] = None,
    out_nm: str = "",
    num_epochs: int = 10,
    batch_size: int = 2048,
    num_workers: int = 2,
    learning_rate: float = 0.0004,
    scheduler_gamma: float = 0.9,
    dropout_rate: float = 0.25,
    resnet_scale: int = 18,
) -> None:
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_prefix = f"{out_nm}_{run_ts}"
    print(f"Train {out_nm!r} → {checkpoint_prefix}_epoch_*.pth")

    fov_dir_path_list_train = list()
    fov_dir_path_list_val = list()
    fov_num_per_project_train = list()
    fov_num_per_project_val = list()
    
    for i, path_to_mantis_dir in enumerate(path_to_mantis_dirs):
        fov_dir_path_list = list_fov_dirs_with_segmentation(path_to_mantis_dir)
        fov_dir_path_list_train.extend(list(np.take(fov_dir_path_list, fov_inds_for_training_per_project[i])))

        fov_num_per_project_train.append(len(fov_inds_for_training_per_project[i]))

        val_inds_mask = np.ones(len(fov_dir_path_list))
        val_inds_mask[fov_inds_for_training_per_project[i]] = 0
        val_inds = np.where(val_inds_mask)[0]
        fov_dir_path_list_val.extend(list(np.take(fov_dir_path_list, val_inds)))

        fov_num_per_project_val.append(len(val_inds))

    print(
        f"  FOVs train={len(fov_dir_path_list_train)} val={len(fov_dir_path_list_val)} "
        f"{list(zip(fov_num_per_project_train, fov_num_per_project_val))}"
    )

    label_dict_train, weight_list_train, n_samples_train = parse_label_csv_files(
        label_csv_path_list,
        fov_dir_path_list_train,
        channel_list_to_filter=marker_expression_to_filter,
        verbose=False,
    )

    train_dataset = FOVDatasetByLabelDict(
        manits_img_dir_path_list=fov_dir_path_list_train,
        labels_dict=label_dict_train,
        add_augmentations=True,
        cache_images=False,
        verbose=False,
    )
    if n_samples_train > 0:
        train_sampler_n = max(1, n_samples_train // 10)
    else:
        train_sampler_n = max(1, len(train_dataset) // 10)
        print(
            f"  Warning: class-weight schedule returned n_samples=0; "
            f"using len(train_dataset)//10 = {train_sampler_n} draws/epoch instead."
        )

    sampler_train = WeightedRandomSampler(
        weight_list_train, num_samples=train_sampler_n, replacement=True
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler_train
    )

    label_dict_val, weight_list_val, n_samples_val = parse_label_csv_files(
        label_csv_path_list,
        fov_dir_path_list_val,
        channel_list_to_filter=marker_expression_to_filter,
        verbose=False,
    )

    val_dataset = FOVDatasetByLabelDict(
        manits_img_dir_path_list=fov_dir_path_list_val,
        labels_dict=label_dict_val,
        cache_images=False,
        verbose=False,
    )
    if n_samples_val > 0:
        val_sampler_n = max(1, n_samples_val // 10)
    else:
        val_sampler_n = max(1, len(val_dataset) // 10)
        print(
            f"  Warning: val class-weight n_samples=0; "
            f"using len(val_dataset)//10 = {val_sampler_n} draws/epoch."
        )

    sampler_val = WeightedRandomSampler(
        weight_list_val, num_samples=val_sampler_n, replacement=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler_val
    )

    print(
        f"  samples train={len(train_dataset)} val={len(val_dataset)} | "
        f"batches/epoch {len(train_loader)}/{len(val_loader)} | "
        f"sampler/epoch {train_sampler_n}/{val_sampler_n} | batch={batch_size} workers={num_workers}\n"
    )

    model = ResNet18CellWithSegFeatureExtractor(
        num_classes=1, feature_dim=32, dropout_rate=dropout_rate, resnet_scale=resnet_scale
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_s = (
        f"{device} ({torch.cuda.get_device_name(0)})"
        if device.type == "cuda"
        else str(device)
    )
    model = model.to(device)

    if model_w_path_to_load is not None:
        state_curr = torch.load(model_w_path_to_load, map_location=device)
        model.load_state_dict(state_curr)
        warm = f"warm-start {os.path.basename(model_w_path_to_load)}"
    else:
        warm = "weights: ImageNet backbone"

    tic = time.time()
    model.train()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)
    print(
        f"{dev_s} | ResNet18+seg dropout={dropout_rate} scale={resnet_scale} | "
        f"Adam lr={learning_rate} ExpLR γ={scheduler_gamma} | {warm}\n"
    )

    for epoch in tqdm.tqdm(range(num_epochs), desc="Epoch", position=0, leave=True):
        model.train()
        running_loss = 0.0
        perf_calcultor = PerfCalculator("Train")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for images, labels, cell_data in tqdm.tqdm(
            train_loader,
            desc="train",
            total=len(train_loader),
            leave=False,
            position=1,
            disable=len(train_loader) <= 1,
        ):
            labels = labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()

            outputs, _ = model((images.float().to(device)))

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            predicted = torch.sigmoid(outputs).round()

            perf_calcultor(labels=labels, preds=predicted)


        mean_train_loss = running_loss / max(len(train_loader), 1)
        t_acc, _, _, t_f1 = perf_calcultor.get_perf(print_results=False)

        print(
            f"\nEpoch {epoch + 1}/{num_epochs}  "
            f"train loss={mean_train_loss:.4f} acc={t_acc}% F1={t_f1}%"
        )
        val_loss = eval_model(
            model,
            device,
            val_loader,
            ratio_to_eval=1,
            str2add=f"e{epoch}",
            verbose=False,
        )
        if device.type == "cuda":
            peak_allocated_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
            peak_reserved_gb = torch.cuda.max_memory_reserved(device) / (1024**3)
            print(
                f"  GPU mem peak: alloc {peak_allocated_gb:.2f} GB | reserved {peak_reserved_gb:.2f} GB"
            )

        scheduler.step()
        lrs = scheduler.get_last_lr()
        if lrs:
            print(f"  lr → {lrs[0]:.6g}")

        if path_to_output_dir is None:
            path_to_output_dir = "trained_models"
        is_final_epoch = (epoch + 1) == num_epochs
        save_this_epoch = (epoch - 1) % 2 == 0 or is_final_epoch
        if save_this_epoch:
            path_to_save = os.path.join(
                path_to_output_dir,
                checkpoint_prefix + "_epoch_" + str(epoch + 1) + ".pth",
            )
            torch.save(model.state_dict(), path_to_save)
            tag = "final" if is_final_epoch else "ckpt"
            print(f"  [{tag}] {os.path.basename(path_to_save)}")
            if is_final_epoch:
                try:
                    from scripts.export_model_for_eval import trace_model_from_memory

                    pt_path = path_to_save.replace(".pth", ".pt")
                    trace_model_from_memory(model, pt_path, device=device)
                    print(f"  [{tag}] {os.path.basename(pt_path)}")
                except Exception as e:
                    print(f"  TorchScript skipped: {e}")

    elapsed_min = (time.time() - tic) / 60.0
    print(f"\nDone in {elapsed_min:.2f} min.\n")


def run_training():
    ratio_train = 0.75
    training_project_list = [
        'PDAC_Aug1423', 'NSCLC_Sep1923', 'GVHD_Feb1323', 'Melanoma_Sept1022',
    ]

    path_to_manits_dir_list = list()
    path_to_label_csv_list = list()
    for project_name in training_project_list:
        path_to_manits_dir_list.append(get_mantis_dir_path_from_proj_name(project_name))
        path_to_label_csv_list.append(get_path_to_label_csv(project_name, get_with_filters=True))

    train_indices_dict = create_train_indices(x=ratio_train)
    training_inds_fovs_per_project = [
        train_indices_dict[project_nm] for project_nm in training_project_list
    ]

    out_nm = 'Resnet18_seg_gmpn_by_patient'
    train_using_seg(
        path_to_manits_dir_list,
        path_to_label_csv_list,
        training_inds_fovs_per_project,
        marker_expression_to_filter=None,
        model_w_path_to_load=None,
        out_nm=out_nm,
    )



if __name__ == '__main__':
    run_training()