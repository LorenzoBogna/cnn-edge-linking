from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


def _dense_boundary(boundary):
    if hasattr(boundary, "toarray"):
        boundary = boundary.toarray()
    return np.asarray(boundary, dtype=np.float32)


def load_bsds_ground_truth(mat_path: str) -> np.ndarray:
    """
    Average BSDS500 human boundary annotations into a soft target [0, 1].
    """
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    gt = mat["groundTruth"]

    if not isinstance(gt, np.ndarray):
        gt = np.array([gt], dtype=object)

    boundaries = []

    for ann in gt.flat:
        if hasattr(ann, "Boundaries"):
            boundaries.append(_dense_boundary(ann.Boundaries))
        elif isinstance(ann, np.void) and "Boundaries" in ann.dtype.names:
            boundaries.append(_dense_boundary(ann["Boundaries"]))

    if not boundaries:
        raise RuntimeError(f"Could not parse Boundaries from {mat_path}")

    target = np.mean(boundaries, axis=0)
    return np.clip(target, 0.0, 1.0).astype(np.float32)


def _augment_pair(bgr: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same random geometric transforms to image and edge target,
    plus photometric transforms to the image only.

    Kept deliberately mild to preserve boundary geometry.
    """
    # Horizontal flip.
    if np.random.rand() < 0.5:
        bgr = cv2.flip(bgr, 1)
        target = cv2.flip(target, 1)

    # Mild random rotation. Reflect padding avoids artificial black borders.
    if np.random.rand() < 0.5:
        angle = float(np.random.uniform(-10.0, 10.0))
        h, w = bgr.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        bgr = cv2.warpAffine(
            bgr, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        target = cv2.warpAffine(
            target, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    # Random crop/scale, then resize back to the original sample size.
    if np.random.rand() < 0.5:
        h, w = bgr.shape[:2]
        scale = float(np.random.uniform(0.80, 1.0))
        ch = max(2, int(round(h * scale)))
        cw = max(2, int(round(w * scale)))
        y0 = int(np.random.randint(0, h - ch + 1))
        x0 = int(np.random.randint(0, w - cw + 1))
        bgr = cv2.resize(bgr[y0:y0+ch, x0:x0+cw], (w, h), interpolation=cv2.INTER_LINEAR)
        target = cv2.resize(target[y0:y0+ch, x0:x0+cw], (w, h), interpolation=cv2.INTER_LINEAR)

    # Brightness/contrast jitter on the image only.
    if np.random.rand() < 0.8:
        alpha = float(np.random.uniform(0.85, 1.15))  # contrast
        beta = float(np.random.uniform(-18.0, 18.0))  # brightness
        bgr = np.clip(bgr.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # Occasional mild Gaussian blur.
    if np.random.rand() < 0.15:
        sigma = float(np.random.uniform(0.3, 1.0))
        bgr = cv2.GaussianBlur(bgr, (3, 3), sigmaX=sigma)

    return bgr, np.clip(target, 0.0, 1.0).astype(np.float32)


class BSDS500Edges(Dataset):
    """
    Expected structure:

    BSDS500/
      data/
        images/
          train/*.jpg
          val/*.jpg
          test/*.jpg
        groundTruth/
          train/*.mat
          val/*.mat
          test/*.mat

    CNN input: RGB only.
    Target: human boundary probability map.
    """

    def __init__(self, root, split="train", image_size=(320, 320), augment=False):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.augment = bool(augment)

        self.image_dir = self.root / "data" / "images" / split
        self.gt_dir = self.root / "data" / "groundTruth" / split

        self.images = sorted(self.image_dir.glob("*.jpg"))

        if not self.images:
            raise RuntimeError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        gt_path = self.gt_dir / f"{img_path.stem}.mat"

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise FileNotFoundError(img_path)

        target = load_bsds_ground_truth(str(gt_path))

        if self.augment:
            bgr, target = _augment_pair(bgr, target)

        h, w = self.image_size
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        target = cv2.resize(target, (w, h), interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        x = torch.from_numpy(rgb.transpose(2, 0, 1)).float()
        y = torch.from_numpy(target[None, ...]).float()

        return x, y
