# Minimal, adapted from TNIA tnia/deeplearning/dl_helper.py

from pathlib import Path
import numpy as np
import os

# ---- from original dl_helper: tiny helper used by quantile_normalization ----
def normalize_(img, low, high, eps=1.e-20, clip=True):
    """
    Scale 'img' to [0,1] using provided low/high bounds, with small eps to avoid /0.
    """
    scaled = (img - low) / (high - low + eps)
    if clip:
        scaled = np.clip(scaled, 0, 1)
    return scaled

# ---- from original dl_helper: quantile_normalization ----
def quantile_normalization(img, quantile_low=0.01, quantile_high=0.998,
                           eps=1.e-20, clip=True, channels=False):
    """
    Copying this from PolBias GPU course.... it is an easy piece of code that is also in stardist.
    But... sometimes Stardist isn't installed, sometimes it is, ... PyTorch is there sometimes it isn't.
    So I'm copying it here.

    First scales the data so that values below quantile_low are smaller than 0 and
    values larger than quantile_high are larger than one. Then optionally clips to (0, 1) range.

    Args:
        img: np.ndarray, (H,W) or (H,W,C)
        quantile_low: float
        quantile_high: float
        eps: float
        clip: bool
        channels: bool, if True and img has channels-last, normalize per channel
    Returns:
        np.ndarray in [0,1], same shape as input
    """
    # if the image is 2D set channels to False
    if len(img.shape) == 2:
        channels = False

    if channels == False:
        qlow = np.quantile(img, quantile_low)
        qhigh = np.quantile(img, quantile_high)
        scaled = normalize_(img, low=qlow, high=qhigh, eps=eps, clip=clip)
        return scaled
    else:
        num_channels = img.shape[-1]
        scaled = np.zeros(img.shape, dtype=np.float32)
        for i in range(num_channels):
            qlow = np.quantile(img[..., i], quantile_low)
            qhigh = np.quantile(img[..., i], quantile_high)
            scaled[..., i] = normalize_(img[..., i], low=qlow, high=qhigh, eps=eps, clip=clip)
        return scaled

# ---- from original dl_helper: directory helpers used by get_label_paths ----
def get_patch_directory(num_inputs, num_truths, parent_dir):
    """gets the directory to put patches from an image and its corresponding ground truth

    Args:
        num_inputs (int)
        num_truths (int)
        parent_dir (str)
    Returns:
        (list[str], list[str]) input_paths, truth_paths
    """
    input_paths = []
    truth_paths = []

    for i in range(num_inputs):
        input_path = os.path.join(parent_dir, "input" + str(i))
        input_paths.append(input_path)

    for i in range(num_truths):
        truth_path = os.path.join(parent_dir, "ground truth" + str(i))
        truth_paths.append(truth_path)

    return input_paths, truth_paths

# ---- from original dl_helper: get_label_paths ----
def get_label_paths(num_inputs, num_truths, parent_dir):
    """gets the paths to put labels from an image and its corresponding ground truth

    Note1: Labels are subtly different than patches. Labels for an image and labels in the same
    directory can be different sizes. Patches are always the same size and cropped from the image.

    Note2: Right now naming scheme for labels and patches are the same, but this function is kept
    for completeness.

    Args:
        num_inputs (int)
        num_truths (int)
        parent_dir (str)
    Returns:
        (list[pathlib.Path], list[pathlib.Path]) image_label_paths, ground_truth_label_paths
    """
    image_dirs, label_dirs = get_patch_directory(num_inputs, num_truths, parent_dir)

    image_paths = []
    for p in image_dirs:
        as_path = Path(p)
        image_paths.append(as_path)

    label_paths = []
    for p in label_dirs:
        as_path = Path(p)
        label_paths.append(as_path)

    return image_paths, label_paths

# ===== Shared image/label helpers (importable across scripts) =====

from tifffile import imread as _tf_imread
import imageio.v3 as _iio
import numpy as _np
from pathlib import Path as _Path

def ensure_hwc_1to5(x, debug=False):
    """Standardize to HWC; keep 1..5 channels; drop trivial alpha. No resizing."""
    x = _np.asarray(x)
    if x.ndim == 2:
        return x
    if x.ndim != 3:
        if debug: print(f"[dl_helper] WARN unexpected ndim={x.ndim}; squeeze.")
        return _np.squeeze(x)
    # CHW → HWC if channels-first and H==W
    if x.shape[0] in (1,2,3,4,5) and x.shape[1] == x.shape[2]:
        if debug: print(f"[dl_helper] CHW→HWC: {x.shape}")
        x = _np.moveaxis(x, 0, -1)
    # drop trivial alpha
    if x.shape[-1] == 4:
        a = x[..., 3]
        near_all_255 = (_np.count_nonzero(a > 250) / a.size) > 0.99
        near_all_0   = (_np.count_nonzero(a) / a.size) < 0.01
        if near_all_255 or near_all_0:
            if debug: print("[dl_helper] drop alpha (RGBA→RGB).")
            x = x[..., :3]
    # cap to first 5 channels
    if x.shape[-1] > 5:
        if debug: print(f"[dl_helper] WARN {x.shape[-1]} channels > 5; keep first 5.")
        x = x[..., :5]
    return x

def has_label(lbl_dir, stem, mask_suffix):
    """Paired selection using YAML labels.mask_filter first, then common fallbacks."""
    lbl_dir = _Path(lbl_dir)
    cands = []
    if mask_suffix:
        cands.append(lbl_dir / f"{stem}{mask_suffix}")
    cands += [
        lbl_dir / f"{stem}_masks.tif",
        lbl_dir / f"{stem}.png",
        lbl_dir / f"{stem}.npy",
        lbl_dir / f"{stem}_seg.npy",
    ]
    return any(p.exists() for p in cands)

def read_label(lbl_path, debug=False):
    """Robust label read for .npy / PNG(JPG,BMP) / TIFF; collapses RGB mask to first plane."""
    lbl_path = _Path(lbl_path)
    suf = lbl_path.suffix.lower()
    if suf == ".npy":
        Y = _np.load(lbl_path)
    elif suf in (".png", ".jpg", ".jpeg", ".bmp"):
        if debug: print("[dl_helper] imageio.imread label")
        Y = _iio.imread(lbl_path)
        if Y.ndim == 3:
            if debug: print(f"[dl_helper] label RGB(A) → first plane, shape={Y.shape}")
            Y = Y[..., 0]
    else:
        if debug: print("[dl_helper] tifffile.imread label")
        Y = _tf_imread(lbl_path)
    return Y

def squish_hwC(x, mode="max"):
    """Collapse channels → one plane; keep H×W; return float32 in [0,1]."""
    x = _np.asarray(x)
    if x.ndim == 2:
        out = x.astype(_np.float32)
    else:
        if mode == "max":
            out = x.max(axis=-1)
        elif mode == "mean":
            out = x.mean(axis=-1)
        elif mode == "sum":
            out = x.sum(axis=-1)
        else:
            raise ValueError("mode must be max|mean|sum")
        out = out.astype(_np.float32)
        mx = float(out.max())
        if mx > 1.0:
            out /= (mx if mx > 0 else 1.0)
    return out