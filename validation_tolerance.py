import argparse

import numpy as np
import torch
import torch.nn.functional as F
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
def evaluate_threshold(
    model,
    loader,
    device,
    threshold,
    gt_threshold=0.20,
    tolerance=2,
):
    """
    Calcola precision, recall e F1 per edge detection.

    A differenza del confronto pixel-per-pixel esatto, considera corretta
    una predizione se esiste un pixel di ground truth entro +/- tolerance
    pixel (e viceversa per il recall).

    tolerance=0 -> comportamento equivalente al matching pixel esatto.
    tolerance=1 -> tolleranza di 1 pixel.
    tolerance=2 -> tolleranza di 2 pixel.
    """

    model.eval()

    matched_pred = 0
    total_pred = 0

    matched_gt = 0
    total_gt = 0

    kernel_size = 2 * tolerance + 1

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        prob = torch.sigmoid(logits)

        # Predizione binaria.
        pred = prob >= threshold

        # BSDS ha una ground truth soft, ottenuta mediando le annotazioni.
        gt = y >= gt_threshold

        if tolerance > 0:
            # Dilata la GT: una prediction viene considerata corretta
            # se cade entro "tolerance" pixel da un edge GT.
            gt_dilated = F.max_pool2d(
                gt.float(),
                kernel_size=kernel_size,
                stride=1,
                padding=tolerance,
            ).bool()

            # Dilata la prediction: un pixel GT viene considerato trovato
            # se esiste una prediction entro "tolerance" pixel.
            pred_dilated = F.max_pool2d(
                pred.float(),
                kernel_size=kernel_size,
                stride=1,
                padding=tolerance,
            ).bool()
        else:
            gt_dilated = gt
            pred_dilated = pred

        # Precision:
        # quante prediction hanno almeno un GT vicino.
        matched_pred += (pred & gt_dilated).sum().item()
        total_pred += pred.sum().item()

        # Recall:
        # quanti pixel GT hanno almeno una prediction vicina.
        matched_gt += (gt & pred_dilated).sum().item()
        total_gt += gt.sum().item()

    precision = matched_pred / (total_pred + 1e-8)
    recall = matched_gt / (total_gt + 1e-8)

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

    parser.add_argument(
        "--gt-threshold",
        type=float,
        default=0.20,
        help="Threshold usato per binarizzare la soft ground truth BSDS.",
    )

    parser.add_argument(
        "--tolerance",
        type=int,
        default=2,
        help=(
            "Tolleranza spaziale in pixel per il matching degli edge. "
            "0 = matching esatto, 1 = +/-1 px, 2 = +/-2 px, ecc."
        ),
    )

    args = parser.parse_args()

    if args.tolerance < 0:
        parser.error("--tolerance deve essere >= 0")

    if not 0.0 <= args.gt_threshold <= 1.0:
        parser.error("--gt-threshold deve essere compreso tra 0 e 1")

    device = get_device()
    print("device:", device)
    print("edge matching tolerance:", args.tolerance, "px")
    print("ground-truth threshold:", args.gt_threshold)

    val_ds = BSDS500Edges(
        args.bsds_root,
        split="val",
        image_size=(args.height, args.width),
        augment=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # IMPORTANTE:
    # Deve essere lo stesso modello usato in training.
    #
    #model = TinyEdgeNet(in_channels=3).to(device)
    #model = TinyEdgeNetV2(in_channels=3).to(device)
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
    print(
        f"Validation con tolerance={args.tolerance}px "
        f"e gt_threshold={args.gt_threshold:.2f}"
    )
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
            gt_threshold=args.gt_threshold,
            tolerance=args.tolerance,
        )

        print(
            f"{threshold:.2f}    "
            f"{precision:.4f}      "
            f"{recall:.4f}    "
            f"{f1:.4f}"
        )

        if best is None or f1 > best["f1"]:
            best = {
                "threshold": float(threshold),
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
        f"F1={best['f1']:.4f} "
        f"tolerance={args.tolerance}px"
    )


if __name__ == "__main__":
    main()
