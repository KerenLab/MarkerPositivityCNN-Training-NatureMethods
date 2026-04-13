import torch
import torch.nn as nn
import tqdm

from data_tools.channels_data import MarkerType
from evaluation.perf_calculator import PerfCalculator


def eval_model(
    model,
    device,
    test_loader,
    th=0.5,
    ratio_to_eval=1,
    str2add="",
    *,
    verbose: bool = True,
):
    """If ``verbose`` is False, prints one validation summary line (for training loops)."""
    criterion = nn.BCEWithLogitsLoss()

    model.eval()
    total_perf_calc = PerfCalculator(f"Eval {str2add}: All markers")
    nuclear_perf_calc = PerfCalculator(f"Eval {str2add}: Nuclear markers")
    cyto_perf_calc = PerfCalculator(f"Eval {str2add}: Cytoplasmic markers")
    membranal_perf_calc = PerfCalculator(f"Eval {str2add}: Membranal markers")

    num_batch_to_eval = len(test_loader) * ratio_to_eval
    cur_batch = 0
    tot_loss = 0
    with torch.no_grad():
        for images, labels, cell_data in tqdm.tqdm(
            test_loader,
            desc="Eval batches",
            total=len(test_loader),
            leave=False,
            disable=not verbose,
        ):
            if cur_batch > num_batch_to_eval:
                break
            cur_batch += 1
            _, _, _, expression = cell_data
            if model.__class__.__name__ == "EfficientNet":
                outputs = model((images.float().to(device)))
            else:
                outputs, _ = model((images.float().to(device)))
            loss = criterion(outputs, labels.float().unsqueeze(1).to(device))
            tot_loss += loss.item()

            indices_membranal = torch.nonzero(
                (expression == MarkerType.Membranal) & (labels != -1), as_tuple=False
            ).squeeze()
            indices_cyto = torch.nonzero(
                (expression == MarkerType.Cytoplasmic) & (labels != -1), as_tuple=False
            ).squeeze()
            indices_nuclear = torch.nonzero(
                (expression == MarkerType.Nuclear) & (labels != -1), as_tuple=False
            ).squeeze()

            predicted = torch.sigmoid(outputs) >= th
            inds_to_get = torch.nonzero(labels != -1)
            if len(labels[inds_to_get].size()) == 0:
                continue
            labels = labels.to(device)
            total_perf_calc(predicted[inds_to_get], labels[inds_to_get])

            if (indices_nuclear.dim() != 0) and (len(indices_nuclear) > 0):
                nuclear_perf_calc(predicted[indices_nuclear], labels[indices_nuclear])
            if (indices_cyto.dim() != 0) and (len(indices_cyto) > 0):
                cyto_perf_calc(predicted[indices_cyto], labels[indices_cyto])
            if (indices_membranal.dim() != 0) and (len(indices_membranal) > 0):
                membranal_perf_calc(predicted[indices_membranal], labels[indices_membranal])

    mean_loss = tot_loss / len(test_loader)
    if verbose:
        total_perf_calc.get_perf(print_results=True)
        nuclear_perf_calc.get_perf(print_results=True)
        cyto_perf_calc.get_perf(print_results=True)
        membranal_perf_calc.get_perf(print_results=True)
    else:
        acc, _, _, f1 = total_perf_calc.get_perf(print_results=False)
        nuclear_perf_calc.get_perf(print_results=False)
        cyto_perf_calc.get_perf(print_results=False)
        membranal_perf_calc.get_perf(print_results=False)
        print(f"  val: loss={mean_loss:.4f} | all-markers acc={acc}% F1={f1}%")

    return mean_loss
