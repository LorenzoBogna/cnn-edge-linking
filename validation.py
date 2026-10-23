import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_bsds import BSDS500Edges
from model import TinyEdgeNet, TinyEdgeNetV2, TinyEdgeNetV3


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate_threshold(model, loader, device, threshold, gt_threshold=0.20):
    model.eval()

    tp = 0
    fp = 0
    fn = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        prob = torch.sigmoid(logits)

        pred = prob >= threshold

        # Ground truth BSDS è soft: la rendiamo binaria
        gt = y >= gt_threshold

        tp += (pred & gt).sum().item()
        fp += (pred & ~gt).sum().item()
        fn += (~pred & gt).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    f1 = 2 * precision * recall / (
        precision + recall + 1e-8
    )

    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bsds-root", required=True)
    parser.add_argument("--weights", required=True)

    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=320)

    parser.add_argument("--batch-size", type=int, default=8)

    args = parser.parse_args()

    device = get_device()
    print("device:", device)

    val_ds = BSDS500Edges(
        args.bsds_root,
        split="val",
        image_size=(args.height, args.width),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # model = TinyEdgeNet(in_channels=3).to(device)
    model = TinyEdgeNetV3(in_channels=3).to(device)

    checkpoint = torch.load(
        args.weights,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model"])
    model.eval()

    print("checkpoint epoch:", checkpoint.get("epoch"))
    print("checkpoint val_loss:", checkpoint.get("val_loss"))

    print()
    print("thr     precision   recall      F1")
    print("-----------------------------------")

    best = None

    thresholds = np.arange(0.05, 0.95, 0.05)

    for threshold in thresholds:
        precision, recall, f1 = evaluate_threshold(
            model,
            val_loader,
            device,
            threshold=float(threshold),
        )

        print(
            f"{threshold:.2f}    "
            f"{precision:.4f}      "
            f"{recall:.4f}    "
            f"{f1:.4f}"
        )

        if best is None or f1 > best["f1"]:
            best = {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

    print()
    print("BEST")
    print(
        f"threshold={best['threshold']:.2f} "
        f"precision={best['precision']:.4f} "
        f"recall={best['recall']:.4f} "
        f"F1={best['f1']:.4f}"
    )


if __name__ == "__main__":
    main()