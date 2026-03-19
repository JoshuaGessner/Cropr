#!/usr/bin/env python3
"""
Cropr - Intelligent batch image cropping.

Crops sets of images to a user-specified aspect ratio and/or resolution,
using subject detection to keep the most important content in frame.
"""

import argparse
import glob as glob_module
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Supported input image extensions
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ---------------------------------------------------------------------------
# Subject detection
# ---------------------------------------------------------------------------

def _cascade_path(filename: str) -> str:
    """Return the absolute path to a bundled OpenCV Haar cascade XML file."""
    return os.path.join(cv2.data.haarcascades, filename)


def _bounding_box(detections):
    """Return a single bounding box (x, y, w, h) enclosing all *detections*."""
    x_min = int(min(d[0] for d in detections))
    y_min = int(min(d[1] for d in detections))
    x_max = int(max(d[0] + d[2] for d in detections))
    y_max = int(max(d[1] + d[3] for d in detections))
    return x_min, y_min, x_max - x_min, y_max - y_min


def detect_subject_face(gray: np.ndarray):
    """
    Attempt to detect faces (frontal + profile) and upper bodies.

    Returns a (x, y, w, h) bounding box that encloses every detected region,
    or ``None`` when nothing is found.
    """
    cascades = [
        ("haarcascade_frontalface_default.xml", dict(scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))),
        ("haarcascade_profileface.xml",         dict(scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))),
        ("haarcascade_upperbody.xml",            dict(scaleFactor=1.05, minNeighbors=3, minSize=(60, 60))),
        ("haarcascade_fullbody.xml",             dict(scaleFactor=1.05, minNeighbors=3, minSize=(60, 60))),
    ]

    found = []
    for xml_file, kwargs in cascades:
        path = _cascade_path(xml_file)
        if not os.path.isfile(path):
            continue
        cascade = cv2.CascadeClassifier(path)
        detections = cascade.detectMultiScale(gray, **kwargs)
        if len(detections) > 0:
            found.extend(detections)
        # Stop after the first cascade that yields results so we do not mix
        # different detector types into the same bounding box.
        if found:
            break

    if not found:
        return None

    return _bounding_box(found)


def detect_subject_entropy(gray: np.ndarray):
    """
    Estimate the most "interesting" region using local entropy (texture).

    Divides the image into a coarse grid, scores each cell by its gradient
    magnitude, and returns the bounding box of the highest-scoring region.
    This acts as a lightweight saliency fallback when no face/body is found.

    Returns a (x, y, w, h) bounding box or ``None``.
    """
    h, w = gray.shape
    if h == 0 or w == 0:
        return None

    grid_rows, grid_cols = 5, 5
    cell_h = h // grid_rows
    cell_w = w // grid_cols

    if cell_h == 0 or cell_w == 0:
        return None

    # Compute Laplacian (edge energy) per grid cell
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    energy = np.abs(lap)

    scores = np.zeros((grid_rows, grid_cols), dtype=float)
    for r in range(grid_rows):
        for c in range(grid_cols):
            cell = energy[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
            scores[r, c] = cell.mean()

    best = np.unravel_index(np.argmax(scores), scores.shape)
    br, bc = best

    # Expand to a 3×3 neighbourhood of cells (clamped to grid)
    r0 = max(0, br - 1)
    r1 = min(grid_rows - 1, br + 1)
    c0 = max(0, bc - 1)
    c1 = min(grid_cols - 1, bc + 1)

    x = c0 * cell_w
    y = r0 * cell_h
    region_w = (c1 + 1) * cell_w - x
    region_h = (r1 + 1) * cell_h - y

    return x, y, region_w, region_h


def detect_subject(image_bgr: np.ndarray, method: str = "face"):
    """
    Detect the primary subject in *image_bgr*.

    Parameters
    ----------
    image_bgr:
        NumPy array in BGR colour order (as returned by ``cv2.imread``).
    method:
        ``"face"``    – Haar-cascade face/body detection with entropy fallback.
        ``"entropy"`` – Entropy/gradient-based region detection only.
        ``"center"``  – Always return ``None`` (forces centre crop).

    Returns
    -------
    (x, y, w, h) bounding box or ``None``.
    """
    if method == "center":
        return None

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    if method == "face":
        box = detect_subject_face(gray)
        if box is not None:
            return box
        # Fall back to entropy when no face/body is detected
        return detect_subject_entropy(gray)

    if method == "entropy":
        return detect_subject_entropy(gray)

    return None


# ---------------------------------------------------------------------------
# Cropping geometry helpers
# ---------------------------------------------------------------------------

def crop_to_aspect_ratio(
    image: Image.Image,
    target_w: float,
    target_h: float,
    subject_box=None,
) -> Image.Image:
    """
    Crop *image* so its aspect ratio matches *target_w* : *target_h*.

    The crop window is centred on the subject when *subject_box* is provided,
    otherwise it is centred on the image.  The window is always clamped so it
    stays within the original image boundaries.
    """
    img_w, img_h = image.size
    target_ratio = target_w / target_h
    current_ratio = img_w / img_h

    if abs(current_ratio - target_ratio) < 1e-6:
        return image  # already the right ratio

    if current_ratio > target_ratio:
        # Too wide – reduce width
        new_w = int(round(img_h * target_ratio))
        new_h = img_h
    else:
        # Too tall – reduce height
        new_w = img_w
        new_h = int(round(img_w / target_ratio))

    # Determine the anchor point (centre of the region of interest)
    if subject_box is not None:
        sx, sy, sw, sh = subject_box
        cx = sx + sw / 2
        cy = sy + sh / 2
    else:
        cx = img_w / 2
        cy = img_h / 2

    left = int(cx - new_w / 2)
    top = int(cy - new_h / 2)

    # Clamp so the crop window stays inside the image
    left = max(0, min(left, img_w - new_w))
    top = max(0, min(top, img_h - new_h))

    return image.crop((left, top, left + new_w, top + new_h))


# ---------------------------------------------------------------------------
# Per-image pipeline
# ---------------------------------------------------------------------------

def process_image(
    input_path: str,
    output_path: str,
    aspect_ratio=None,
    resolution=None,
    method: str = "face",
) -> None:
    """
    Load, (optionally) crop, (optionally) resize, and save one image.

    Parameters
    ----------
    input_path:
        Path to the source image.
    output_path:
        Destination path (including filename and extension).
    aspect_ratio:
        ``(width, height)`` floats describing the desired crop ratio, or
        ``None`` to skip aspect-ratio cropping.
    resolution:
        ``(width, height)`` ints describing the output pixel dimensions, or
        ``None`` to skip resizing.
    method:
        Subject detection method – ``"face"``, ``"entropy"``, or ``"center"``.
    """
    pil_image = Image.open(input_path).convert("RGB")

    # Determine effective aspect ratio for cropping
    if aspect_ratio is not None:
        ar_w, ar_h = aspect_ratio
    elif resolution is not None:
        ar_w, ar_h = resolution
    else:
        ar_w, ar_h = None, None

    if ar_w is not None:
        # Run subject detection only when we actually need to crop
        if method != "center":
            img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            subject_box = detect_subject(img_bgr, method)
        else:
            subject_box = None

        pil_image = crop_to_aspect_ratio(pil_image, ar_w, ar_h, subject_box)

    # Resize to exact resolution if requested
    if resolution is not None:
        pil_image = pil_image.resize(resolution, Image.LANCZOS)

    # Ensure the output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    pil_image.save(output_path)


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------

def parse_aspect_ratio(ratio_str: str):
    """Parse ``'16:9'``, ``'16x9'``, or ``'16/9'`` into ``(16.0, 9.0)``."""
    for sep in (":", "x", "X", "/"):
        if sep in ratio_str:
            left, right = ratio_str.split(sep, 1)
            try:
                w, h = float(left), float(right)
            except ValueError:
                break
            if w <= 0 or h <= 0:
                raise ValueError(f"Aspect ratio values must be positive, got: {ratio_str!r}")
            return w, h
    raise ValueError(
        f"Invalid aspect ratio {ratio_str!r}. "
        "Use formats like '16:9', '4:3', or '1:1'."
    )


def parse_resolution(res_str: str):
    """Parse ``'1920x1080'`` or ``'1920X1080'`` into ``(1920, 1080)``."""
    for sep in ("x", "X"):
        if sep in res_str:
            left, right = res_str.split(sep, 1)
            try:
                w, h = int(left), int(right)
            except ValueError:
                break
            if w <= 0 or h <= 0:
                raise ValueError(f"Resolution values must be positive integers, got: {res_str!r}")
            return w, h
    raise ValueError(
        f"Invalid resolution {res_str!r}. "
        "Use formats like '1920x1080' or '512x512'."
    )


def collect_input_files(input_arg: str):
    """
    Return a sorted list of :class:`pathlib.Path` objects for all image files
    matched by *input_arg*.

    *input_arg* may be:
    * a directory  – all images inside (non-recursive) are returned,
    * a glob string – all matching paths are returned.
    """
    p = Path(input_arg)
    if p.is_dir():
        files = [
            f for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]
    else:
        files = [
            Path(f) for f in glob_module.glob(input_arg)
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS
        ]
    return sorted(files)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cropr",
        description="Cropr – intelligent batch image cropping.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Crop all JPEGs in ./photos to 16:9 aspect ratio (face-aware)
  cropr.py -i ./photos -o ./output -a 16:9

  # Crop and resize to exactly 512×512 pixels
  cropr.py -i ./photos -o ./output -r 512x512

  # Combine aspect-ratio crop with target resolution
  cropr.py -i ./photos -o ./output -a 1:1 -r 512x512

  # Use simple centre-crop (no detection)
  cropr.py -i ./photos -o ./output -a 4:3 --method center

  # Process a glob pattern and convert to PNG
  cropr.py -i "./raw/**/*.jpg" -o ./output -a 16:9 --format png
""",
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        metavar="PATH",
        help="Input directory or glob pattern.",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        metavar="DIR",
        help="Output directory (created if it does not exist).",
    )
    parser.add_argument(
        "-a", "--aspect-ratio",
        metavar="W:H",
        help="Target aspect ratio, e.g. '16:9', '4:3', '1:1'.",
    )
    parser.add_argument(
        "-r", "--resolution",
        metavar="WxH",
        help="Target output resolution in pixels, e.g. '1920x1080'.",
    )
    parser.add_argument(
        "-m", "--method",
        default="face",
        choices=["face", "entropy", "center"],
        help=(
            "Subject detection method used to guide the crop window. "
            "'face' uses Haar-cascade face/body detection with an entropy "
            "fallback; 'entropy' uses gradient-based saliency only; "
            "'center' always crops from the centre. (default: face)"
        ),
    )
    parser.add_argument(
        "-f", "--format",
        metavar="EXT",
        help=(
            "Force output image format, e.g. 'jpg' or 'png'. "
            "When omitted, the source format is preserved."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files. By default they are skipped.",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # ---- Validate / parse crop parameters --------------------------------
    aspect_ratio = None
    if args.aspect_ratio:
        try:
            aspect_ratio = parse_aspect_ratio(args.aspect_ratio)
        except ValueError as exc:
            parser.error(str(exc))

    resolution = None
    if args.resolution:
        try:
            resolution = parse_resolution(args.resolution)
        except ValueError as exc:
            parser.error(str(exc))

    if aspect_ratio is None and resolution is None:
        parser.error("At least one of --aspect-ratio (-a) or --resolution (-r) is required.")

    # ---- Discover input files --------------------------------------------
    input_files = collect_input_files(args.input)
    if not input_files:
        print(f"No image files found matching: {args.input}", file=sys.stderr)
        return 1

    # ---- Set up output directory -----------------------------------------
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Process each file -----------------------------------------------
    success = skipped = failed = 0

    for img_path in input_files:
        suffix = f".{args.format.lstrip('.')}" if args.format else img_path.suffix
        out_path = output_dir / f"{img_path.stem}{suffix}"

        if out_path.exists() and not args.overwrite:
            print(f"  skip  {img_path.name}  (already exists; use --overwrite)")
            skipped += 1
            continue

        try:
            print(f"  crop  {img_path.name}", end="", flush=True)
            process_image(
                str(img_path),
                str(out_path),
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                method=args.method,
            )
            print(f"  →  {out_path.name}")
            success += 1
        except Exception as exc:  # pylint: disable=broad-except
            print(f"\n  ERROR  {img_path.name}: {exc}", file=sys.stderr)
            failed += 1

    # ---- Summary ---------------------------------------------------------
    print(f"\nDone: {success} processed, {skipped} skipped, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
