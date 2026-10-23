import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from canny_linked import (
    color_contours,
    extract_contours,
    link_cnn_edges,
)
from model import TinyEdgeNet, TinyEdgeNetV2, TinyEdgeNetV3

# ==============================================================================
# PARAMETRI DI DEFAULT E CONFIGURAZIONE
# ==============================================================================
DEFAULT_OUT_DIR = "output"
DEFAULT_HEIGHT = 320
DEFAULT_WIDTH = 320

DEFAULT_LOW_PROB = 0.20
DEFAULT_HIGH_PROB = 0.50
DEFAULT_MIN_GRADIENT = 0.05
DEFAULT_MAX_ANGLE = 10.0
DEFAULT_MAX_GAP = 8

'con gap 10 e angolo 15 -> Tempo totale di elaborazione: 00:00:19 (19.395 secondi) Tempo medio per immagine: 0.924 secondi'
'con gap 8 e angolo 10 -> Tempo totale di elaborazione: 00:00:13 (13.082 secondi) Tempo medio per immagine: 0.623 secondi'


DEFAULT_MIN_AREA = 5
DEFAULT_THICKNESS = 2
# ==============================================================================


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_rgb_input(bgr, size):
    h, w = size

    resized = cv2.resize(
        bgr,
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )

    rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB,
    ).astype(np.float32) / 255.0

    x = torch.from_numpy(
        rgb.transpose(2, 0, 1)
    ).float().unsqueeze(0)

    return x, resized


def process_image(image_path, model, device, args, out_dir):
    bgr = cv2.imread(str(image_path))

    if bgr is None:
        print(f"Errore: impossibile leggere l'immagine {image_path}")
        return

    x, resized = build_rgb_input(
        bgr,
        (args.height, args.width),
    )

    x = x.to(device)

    with torch.no_grad():
        for _ in range(3):
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    if device.type == "cuda":
        torch.cuda.synchronize()

    cnn_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()

    linked_result = link_cnn_edges(
        image_bgr=resized,
        edge_prob=prob,
        low_prob=args.low_prob,
        high_prob=args.high_prob,
        min_gradient=args.min_gradient,
        max_angle_diff_deg=args.max_angle,
        bridge_gaps=True,
        max_gap=args.max_gap,
    )

    linking_ms = (time.perf_counter() - t1) * 1000.0

    linked = linked_result["linked_edges"]

    contours = extract_contours(
        linked,
        min_area=DEFAULT_MIN_AREA,
    )

    colored = color_contours(
        resized,
        contours,
        thickness=DEFAULT_THICKNESS,
    )

    file_out_dir = out_dir if args.is_single_file else (out_dir / image_path.stem)
    file_out_dir.mkdir(parents=True, exist_ok=True)

    prob_img = np.clip(prob * 255.0, 0, 255).astype(np.uint8)

    raw_threshold = (
        prob >= args.high_prob
    ).astype(np.uint8) * 255

    linked_img = linked * 255

    cv2.imwrite(
        str(file_out_dir / "01_cnn_probability.png"),
        prob_img,
    )

    cv2.imwrite(
        str(file_out_dir / "02_cnn_threshold.png"),
        raw_threshold,
    )

    cv2.imwrite(
        str(file_out_dir / "03_linked_edges.png"),
        linked_img,
    )

    cv2.imwrite(
        str(file_out_dir / "04_colored_contours.png"),
        colored,
    )

    print(f"[{image_path.name}] CNN forward: {cnn_ms:.3f} ms | Linking: {linking_ms:.3f} ms | Contours: {len(contours)}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("image", help="Percorso a un file immagine o a una cartella di immagini")
    parser.add_argument("--weights", required=True)

    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)

    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)

    parser.add_argument("--low-prob", type=float, default=DEFAULT_LOW_PROB)
    parser.add_argument("--high-prob", type=float, default=DEFAULT_HIGH_PROB)

    parser.add_argument("--min-gradient", type=float, default=DEFAULT_MIN_GRADIENT)
    parser.add_argument("--max-angle", type=float, default=DEFAULT_MAX_ANGLE)

    parser.add_argument("--max-gap", type=int, default=DEFAULT_MAX_GAP)

    args = parser.parse_args()

    device = get_device()
    print("device:", device)

    # model = TinyEdgeNet(in_channels=3).to(device)
    model = TinyEdgeNetV3(in_channels=3).to(device)

    ckpt = torch.load(
        args.weights,
        map_location=device,
    )

    model.load_state_dict(ckpt["model"])
    model.eval()

    target_path = Path(args.image)
    out_dir = Path(args.out_dir)

    if not target_path.exists():
        raise FileNotFoundError(target_path)

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    if target_path.is_file():
        args.is_single_file = True
        images = [target_path]
    elif target_path.is_dir():
        args.is_single_file = False
        images = [p for p in target_path.iterdir() if p.suffix.lower() in valid_extensions]
        images.sort()
    else:
        raise ValueError(f"Percorso non valido: {target_path}")

    print(f"Trovate {len(images)} immagini da elaborare.")

    # Timer complessivo del batch
    total_start = time.perf_counter()

    for index, img_path in enumerate(images, start=1):
        image_start = time.perf_counter()

        process_image(img_path, model, device, args, out_dir)

        image_elapsed = time.perf_counter() - image_start
        total_elapsed = time.perf_counter() - total_start

        # Stima del tempo rimanente sulla media delle immagini già elaborate
        avg_time = total_elapsed / index
        remaining_images = len(images) - index
        eta_seconds = avg_time * remaining_images

        def format_time(seconds):
            seconds = max(0, int(seconds))
            hours, remainder = divmod(seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        print(
            f"[TIMER {index}/{len(images)}] "
            f"Immagine: {image_elapsed:.3f} s | "
            f"Trascorso: {format_time(total_elapsed)} | "
            f"Rimanente stimato: {format_time(eta_seconds)}"
        )

    total_elapsed = time.perf_counter() - total_start

    print(f"Tempo totale di elaborazione: {format_time(total_elapsed)} "
          f"({total_elapsed:.3f} secondi)")

    if images:
        print(f"Tempo medio per immagine: {total_elapsed / len(images):.3f} secondi")

    print(f"Tutti gli output sono stati salvati in: {out_dir}")


if __name__ == "__main__":
    main()
    
    
    
    '''
    threshold=0.40 precision=0.3680 recall=0.4338 F1=0.3982
    
    '''
    