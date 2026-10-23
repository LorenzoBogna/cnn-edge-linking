import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_bsds import BSDS500Edges
from model import TinyEdgeNetV2, TinyEdgeNetV3
from model import TinyEdgeNet
import sys 

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def soft_dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)

    inter = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def edge_loss(logits, targets, pos_weight=4.0):
    """
    Edge pixels are sparse, so use weighted BCE + Dice.
    """
    weight = torch.tensor([pos_weight], device=logits.device)

    bce = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=weight,
    )

    dice = soft_dice_loss(logits, targets)
    return bce + 0.5 * dice


@torch.no_grad()
def evaluate(model, loader, device, pos_weight=4.0):
    model.eval()

    total = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = edge_loss(logits, y, pos_weight=pos_weight)

        total += loss.item() * x.size(0)
        n += x.size(0)

    return total / max(n, 1)


def _to_uint8_gray(a):
    return np.clip(a * 255.0, 0, 255).astype(np.uint8)


@torch.no_grad()
def save_overfit_diagnostics(model, x, y, device, out_dir, step, threshold):
    """Save input, ground truth and current CNN prediction for one sample."""
    model.eval()

    logits = model(x.to(device))
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    gt = y[0, 0].cpu().numpy()

    rgb = x[0].cpu().numpy().transpose(1, 2, 0)
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    gt_img = _to_uint8_gray(gt)
    prob_img = _to_uint8_gray(prob)
    threshold_img = (prob >= threshold).astype(np.uint8) * 255

    step_dir = out_dir / f"step_{step:04d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(step_dir / "input.png"), bgr)
    cv2.imwrite(str(step_dir / "ground_truth.png"), gt_img)
    cv2.imwrite(str(step_dir / "probability.png"), prob_img)
    cv2.imwrite(str(step_dir / "threshold.png"), threshold_img)

    # Four-panel image for quick visual inspection.
    gt_bgr = cv2.cvtColor(gt_img, cv2.COLOR_GRAY2BGR)
    prob_bgr = cv2.cvtColor(prob_img, cv2.COLOR_GRAY2BGR)
    thr_bgr = cv2.cvtColor(threshold_img, cv2.COLOR_GRAY2BGR)
    panel = np.concatenate([bgr, gt_bgr, prob_bgr, thr_bgr], axis=1)
    cv2.imwrite(str(step_dir / "panel_input_gt_prob_threshold.png"), panel)

    mae = float(np.mean(np.abs(prob - gt)))
    stats = {
        "min": float(prob.min()),
        "max": float(prob.max()),
        "mean": float(prob.mean()),
        "p50": float(np.percentile(prob, 50)),
        "p90": float(np.percentile(prob, 90)),
        "p95": float(np.percentile(prob, 95)),
        "p99": float(np.percentile(prob, 99)),
        "mae_to_soft_gt": mae,
    }

    return stats


def run_overfit_one(model, train_ds, device, args, out_dir):
    """
    Diagnostic mode: repeatedly train on exactly one BSDS500 sample.

    A healthy training pipeline should be able to strongly fit one sample.
    This separates optimization/data-pipeline problems from generalization issues.
    """
    # Cerca l'indice dell'immagine basandosi sul nome passato (es. "2092.jpg")
    overfit_idx = None
    for idx in range(len(train_ds)):
        # Adatta a seconda di come il dataset espone il percorso (es. train_ds.images o train_ds.image_paths)
        # Di solito nei dataset BSDS custom la lista dei percorsi è salvata in .images o .image_paths
        img_path = str(train_ds.images[idx]) 
        if args.overfit_image in img_path:
            overfit_idx = idx
            break

    if overfit_idx is None:
        raise ValueError(
            f"Impossibile trovare l'immagine '{args.overfit_image}' nel dataset di training. "
            f"Verifica che il nome sia corretto."
        )

    x, y = train_ds[overfit_idx]
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)

    x_device = x.to(device)
    y_device = y.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    diag_dir = out_dir / "overfit_one"
    diag_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"OVERFIT-ONE | sample_index={overfit_idx} (file: {args.overfit_image}) | "
        f"steps={args.overfit_steps} | save_every={args.save_every}"
    )

    stats = save_overfit_diagnostics(
        model, x, y, device, diag_dir, step=0, threshold=args.diag_threshold
    )
    print(
        "step 0000 | "
        f"prob[min={stats['min']:.3f} max={stats['max']:.3f} "
        f"mean={stats['mean']:.3f} p95={stats['p95']:.3f} "
        f"p99={stats['p99']:.3f}] | "
        f"MAE={stats['mae_to_soft_gt']:.4f}"
    )

    for step in range(1, args.overfit_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(x_device)
        loss = edge_loss(logits, y_device, pos_weight=args.pos_weight)
        loss.backward()
        optimizer.step()

        should_save = (
            step == 1
            or step % args.save_every == 0
            or step == args.overfit_steps
        )

        if should_save:
            stats = save_overfit_diagnostics(
                model,
                x,
                y,
                device,
                diag_dir,
                step=step,
                threshold=args.diag_threshold,
            )
            print(
                f"step {step:04d} | loss={loss.item():.6f} | "
                f"prob[min={stats['min']:.3f} max={stats['max']:.3f} "
                f"mean={stats['mean']:.3f} p95={stats['p95']:.3f} "
                f"p99={stats['p99']:.3f}] | "
                f"MAE={stats['mae_to_soft_gt']:.4f}"
            )

    checkpoint = {
        "model": model.state_dict(),
        "steps": args.overfit_steps,
        "overfit_index": overfit_idx,
        "overfit_image": args.overfit_image,
        "mode": "overfit_one",
    }
    torch.save(checkpoint, diag_dir / "overfit_last.pt")

    print(f"Diagnostics saved in: {diag_dir}")
    print("Panel order: input | soft ground truth | probability | threshold")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bsds-root", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, default=4.0)

    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=320)

    parser.add_argument("--out", default="checkpoints")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable conservative augmentation (horizontal flip + mild brightness/contrast) for training only.",
    )

    # Diagnostic mode: deliberately memorize one training image.
    parser.add_argument(
        "--overfit-one",
        action="store_true",
        help="Train repeatedly on one BSDS500 training sample for diagnostics.",
    )
    parser.add_argument(
        "--overfit-index",
        type=int,
        default=0,
        help="Training sample index used by --overfit-one.",
    )
    parser.add_argument(
        "--overfit-steps",
        type=int,
        default=1000,
        help="Number of optimization steps in --overfit-one mode.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Save overfit diagnostics every N optimization steps.",
    )
    parser.add_argument(
        "--diag-threshold",
        type=float,
        default=0.20,
        help="Threshold used only for diagnostic threshold images.",
    )

    args = parser.parse_args()

    if args.save_every <= 0:
        parser.error("--save-every must be > 0")
    if args.overfit_steps <= 0:
        parser.error("--overfit-steps must be > 0")
    if not 0.0 <= args.diag_threshold <= 1.0:
        parser.error("--diag-threshold must be between 0 and 1")
    if args.pos_weight <= 0:
        parser.error("--pos-weight must be > 0")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("device:", device)

    train_ds = BSDS500Edges(
        args.bsds_root,
        split="train",
        image_size=(args.height, args.width),
        augment=args.augment and not args.overfit_one,
    )

    #model = TinyEdgeNet(in_channels=3).to(device)
    
    # model = TinyEdgeNetV2(in_channels=3).to(device)
    model = TinyEdgeNetV3(in_channels=3).to(device)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("augmentation:", "on" if (args.augment and not args.overfit_one) else "off")

    if args.overfit_one:
        run_overfit_one(model, train_ds, device, args, out_dir)
        return

    val_ds = BSDS500Edges(
        args.bsds_root,
        split="val",
        image_size=(args.height, args.width),
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()

        total = 0.0
        n = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = edge_loss(logits, y, pos_weight=args.pos_weight)

            loss.backward()
            optimizer.step()

            total += loss.item() * x.size(0)
            n += x.size(0)

        train_loss = total / max(n, 1)
        val_loss = evaluate(
            model,
            val_loader,
            device,
            pos_weight=args.pos_weight,
        )

        print(
            f"epoch {epoch:03d} | "
            f"train_loss={train_loss:.5f} | "
            f"val_loss={val_loss:.5f}"
        )

        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
        }

        torch.save(checkpoint, out_dir / "last.pt")

        if val_loss < best:
            best = val_loss
            torch.save(checkpoint, out_dir / "best.pt")
            print("saved best.pt")


if __name__ == "__main__":
    main()

'''
NB cambia modello in validation, train, infer

python ./cnn/train.py \
  --bsds-root BSDS500 \
  --epochs 30 \
  --pos-weight 4 \
  --out ./output_list/checkpoints_v2

'''
