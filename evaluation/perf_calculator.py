"""Minimal metrics accumulator used by training and evaluation loops."""

from __future__ import annotations

import torch


class PerfCalculator:
    def __init__(self, print_nm: str = "") -> None:
        self.correct = 0
        self.true_pos = 0
        self.false_pos = 0
        self.total_pos = 0
        self.total = 0
        self.print_nm = print_nm

    def __call__(self, preds, labels) -> None:
        if type(preds) == torch.Tensor:
            preds = preds.squeeze().long()
            labels = labels.squeeze()
            self.correct += ((preds == labels) & (labels != -1)).sum().item()
            self.true_pos += ((preds == 1) & (labels == 1)).sum().item()
            self.false_pos += ((preds == 1) & (labels == 0)).sum().item()
            self.total_pos += ((labels == 1)).sum().item()
            self.total += (labels != -1).sum().item()
        else:
            self.correct += ((preds == labels) & (labels != -1)).sum()
            self.true_pos += ((preds == 1) & (labels == 1)).sum()
            self.false_pos += ((preds == 1) & (labels == 0)).sum()
            self.total_pos += ((labels == 1)).sum()
            self.total += (labels != -1).sum()

    def get_perf(self, print_results: bool = True, print_raw_numbers: bool = False):
        if self.total == 0:
            accuracy = 0
        else:
            accuracy = round(100 * self.correct / self.total, 2)

        if self.total_pos == 0:
            recall = 0
        else:
            recall = round(100 * self.true_pos / self.total_pos, 2)

        if self.true_pos == 0:
            precision = 0
        else:
            precision = round(100 * self.true_pos / (self.true_pos + self.false_pos), 2)

        if recall + precision == 0:
            f1 = 0
        else:
            f1 = round(2 * (recall * precision) / (recall + precision), 2)

        if print_raw_numbers:
            print(
                f"Total: {self.total}, true pos: {self.true_pos}, "
                f"false pos: {self.false_pos}, correct: {self.correct}"
            )

        if print_results:
            print(f"{self.print_nm} Accuracy: {accuracy}%")
            print(f"{self.print_nm} Recall: {recall}%")
            print(f"{self.print_nm} Precision: {precision}%")
            print(f"{self.print_nm} F1: {f1}%")

        return accuracy, recall, precision, f1

    def get_perf_as_dict(self, print_results: bool = False):
        accuracy, recall, precision, f1 = self.get_perf(print_results=print_results)
        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        }
