#!/usr/bin/env python3
"""
Export a trained model to TorchScript so the evaluation repo can load it
without the model architecture code. Run from the main repo.

Usage:
  python scripts/export_model_for_eval.py path/to/weights.pth [output.pt]
  Default output: model.pt in the same dir as the weights.
"""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data_tools.torch_models import ResNet18CellWithSegFeatureExtractor


class _ModelWrapper(torch.nn.Module):
    """Returns only logits for TorchScript export; no architecture exposed."""

    def __init__(self, state_dict_path, device="cpu"):
        super().__init__()
        model = ResNet18CellWithSegFeatureExtractor(
            num_classes=1, feature_dim=32, dropout_rate=0.5
        )
        state = torch.load(state_dict_path, map_location=device)
        if next(iter(state.keys())).startswith("module."):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        model.eval()
        self.model = model

    def forward(self, x):
        logits, _ = self.model((x))
        return logits


def trace_model_from_memory(model, output_path, device="cpu", example_shape=(1, 3, 128, 128)):
    """Trace the current model (e.g. after training) and save as TorchScript .pt.
    Call this from the training script to get a .pt alongside each .pth checkpoint.
    """
    class _WrapLogits(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self._m = m

        def forward(self, x):
            logits, _ = self._m((x))
            return logits

    wrapped = _WrapLogits(model)
    wrapped.eval()
    example = torch.rand(example_shape, device=device)
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, example)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output_path))
    return output_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/export_model_for_eval.py path/to/weights.pth [output.pt]")
        sys.exit(1)
    weights_path = Path(sys.argv[1]).resolve()
    if not weights_path.exists():
        print(f"Error: {weights_path} not found", file=sys.stderr)
        sys.exit(1)
    out_path = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else weights_path.parent / "model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = _ModelWrapper(weights_path, device=device)
    wrapper.eval()

    example = torch.rand(2, 3, 128, 128, device=device)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example)

    traced.save(str(out_path))
    print(f"Saved TorchScript model to {out_path}")


if __name__ == "__main__":
    main()
