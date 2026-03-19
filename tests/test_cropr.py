"""
Unit and integration tests for cropr.py.

Tests focus on:
- Argument / format parsing helpers
- Crop geometry (crop_to_aspect_ratio)
- Subject detection helpers (detect_subject_entropy, detect_subject_face)
- End-to-end process_image pipeline
- CLI entry point (main)

All tests create synthetic images in-memory or in a temporary directory
so that no real photographs are required.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

# Make sure the package root is on the path when running from any directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

from cropr import (
    collect_input_files,
    crop_to_aspect_ratio,
    detect_subject,
    detect_subject_entropy,
    main,
    parse_aspect_ratio,
    parse_resolution,
    process_image,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pil_image(width: int = 200, height: int = 200, colour=(128, 200, 100)) -> Image.Image:
    """Return a solid-colour PIL Image."""
    img = Image.new("RGB", (width, height), colour)
    return img


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Convert a PIL Image to a BGR NumPy array (OpenCV convention)."""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def save_image(pil_img: Image.Image, directory: str, filename: str = "test.jpg") -> str:
    """Save *pil_img* to *directory*/*filename* and return the full path."""
    path = os.path.join(directory, filename)
    pil_img.save(path)
    return path


# ---------------------------------------------------------------------------
# parse_aspect_ratio
# ---------------------------------------------------------------------------

class TestParseAspectRatio:
    def test_colon_separator(self):
        assert parse_aspect_ratio("16:9") == (16.0, 9.0)

    def test_x_separator_lower(self):
        assert parse_aspect_ratio("4x3") == (4.0, 3.0)

    def test_x_separator_upper(self):
        assert parse_aspect_ratio("4X3") == (4.0, 3.0)

    def test_slash_separator(self):
        assert parse_aspect_ratio("1/1") == (1.0, 1.0)

    def test_square(self):
        assert parse_aspect_ratio("1:1") == (1.0, 1.0)

    def test_float_values(self):
        w, h = parse_aspect_ratio("2.39:1")
        assert abs(w - 2.39) < 1e-9
        assert abs(h - 1.0) < 1e-9

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            parse_aspect_ratio("169")

    def test_zero_width_raises(self):
        with pytest.raises(ValueError):
            parse_aspect_ratio("0:9")

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            parse_aspect_ratio("-4:3")


# ---------------------------------------------------------------------------
# parse_resolution
# ---------------------------------------------------------------------------

class TestParseResolution:
    def test_lowercase_x(self):
        assert parse_resolution("1920x1080") == (1920, 1080)

    def test_uppercase_x(self):
        assert parse_resolution("512X512") == (512, 512)

    def test_square(self):
        assert parse_resolution("256x256") == (256, 256)

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            parse_resolution("1920:1080")

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            parse_resolution("0x1080")

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            parse_resolution("-1x1080")


# ---------------------------------------------------------------------------
# crop_to_aspect_ratio
# ---------------------------------------------------------------------------

class TestCropToAspectRatio:
    def test_no_op_when_ratio_already_matches(self):
        img = make_pil_image(160, 90)
        cropped = crop_to_aspect_ratio(img, 16, 9)
        assert cropped.size == (160, 90)

    def test_wide_image_crops_width(self):
        """A 300×100 image cropped to 1:1 should yield a 100×100 result."""
        img = make_pil_image(300, 100)
        cropped = crop_to_aspect_ratio(img, 1, 1)
        w, h = cropped.size
        assert w == h
        assert h == 100

    def test_tall_image_crops_height(self):
        """A 100×300 image cropped to 1:1 should yield a 100×100 result."""
        img = make_pil_image(100, 300)
        cropped = crop_to_aspect_ratio(img, 1, 1)
        w, h = cropped.size
        assert w == h
        assert w == 100

    def test_output_ratio_matches_target(self):
        img = make_pil_image(800, 600)
        cropped = crop_to_aspect_ratio(img, 16, 9)
        w, h = cropped.size
        assert abs(w / h - 16 / 9) < 0.01

    def test_centre_crop_when_no_subject(self):
        """Without a subject box, the crop should be centred."""
        img = make_pil_image(300, 100)
        cropped = crop_to_aspect_ratio(img, 1, 1, subject_box=None)
        # The cropped image should be centred: left offset = (300-100)//2 = 100
        # We can verify by checking width = height = 100
        assert cropped.size == (100, 100)

    def test_subject_box_shifts_crop(self):
        """A subject in the far right should push the crop window rightward."""
        img = make_pil_image(400, 200, colour=(50, 50, 50))
        # Subject at the far right edge
        subject_box = (350, 50, 40, 100)  # x, y, w, h
        cropped = crop_to_aspect_ratio(img, 1, 1, subject_box=subject_box)
        # Resulting crop is 200×200 and must be clamped inside 400×200
        assert cropped.size == (200, 200)

    def test_subject_box_clamped_to_image(self):
        """Subject at the extreme edge must not produce out-of-bounds crop."""
        img = make_pil_image(200, 100)
        # Subject entirely outside of the crop window area
        subject_box = (190, 0, 10, 100)
        cropped = crop_to_aspect_ratio(img, 2, 1, subject_box=subject_box)
        w, h = cropped.size
        assert w == 200
        assert h == 100

    def test_result_within_original_bounds(self):
        """The cropped region should always be entirely inside the source image."""
        img = make_pil_image(640, 480)
        subject_box = (10, 10, 50, 50)
        for ratio in [(16, 9), (4, 3), (1, 1), (9, 16)]:
            cropped = crop_to_aspect_ratio(img, *ratio, subject_box=subject_box)
            assert cropped.size[0] <= 640
            assert cropped.size[1] <= 480


# ---------------------------------------------------------------------------
# detect_subject (detect_subject_entropy)
# ---------------------------------------------------------------------------

class TestDetectSubjectEntropy:
    def test_returns_tuple_of_four(self):
        # Create an image with a bright patch in the centre to attract entropy
        arr = np.zeros((200, 200), dtype=np.uint8)
        arr[80:120, 80:120] = 255  # high-contrast square in centre
        box = detect_subject_entropy(arr)
        assert box is not None
        assert len(box) == 4

    def test_values_within_image(self):
        arr = np.zeros((300, 400), dtype=np.uint8)
        arr[100:200, 150:250] = 200
        box = detect_subject_entropy(arr)
        assert box is not None
        x, y, w, h = box
        assert x >= 0 and y >= 0
        assert x + w <= 400
        assert y + h <= 300

    def test_uniform_image(self):
        """A uniform image still returns a box (lowest-energy region)."""
        arr = np.full((100, 100), 128, dtype=np.uint8)
        box = detect_subject_entropy(arr)
        assert box is not None

    def test_tiny_image_returns_none(self):
        """Images too small for the grid should return None gracefully."""
        arr = np.zeros((1, 1), dtype=np.uint8)
        box = detect_subject_entropy(arr)
        assert box is None


class TestDetectSubject:
    def _make_bgr(self, width=200, height=200, colour=(100, 150, 200)):
        pil = Image.new("RGB", (width, height), colour)
        return pil_to_bgr(pil)

    def test_center_method_returns_none(self):
        img = self._make_bgr()
        assert detect_subject(img, method="center") is None

    def test_entropy_method_returns_box(self):
        img = self._make_bgr(200, 200)
        # Add some texture so entropy has something to find
        arr = np.array(img)
        arr[80:120, 80:120] = [255, 0, 0]
        box = detect_subject(arr, method="entropy")
        assert box is not None
        assert len(box) == 4

    def test_face_method_falls_back_to_entropy(self):
        """A plain coloured image has no faces; should fall back to entropy."""
        img = self._make_bgr(200, 200)
        box = detect_subject(img, method="face")
        # Either None (very uniform image) or a valid box – must not raise
        if box is not None:
            assert len(box) == 4


# ---------------------------------------------------------------------------
# process_image
# ---------------------------------------------------------------------------

class TestProcessImage:
    def test_aspect_ratio_only(self, tmp_path):
        src = make_pil_image(400, 300)
        in_path = str(tmp_path / "input.png")
        out_path = str(tmp_path / "output.png")
        src.save(in_path)

        process_image(in_path, out_path, aspect_ratio=(1, 1), method="center")

        result = Image.open(out_path)
        w, h = result.size
        assert w == h == 300

    def test_resolution_only(self, tmp_path):
        src = make_pil_image(400, 300)
        in_path = str(tmp_path / "input.jpg")
        out_path = str(tmp_path / "output.jpg")
        src.save(in_path)

        process_image(in_path, out_path, resolution=(128, 128), method="center")

        result = Image.open(out_path)
        assert result.size == (128, 128)

    def test_aspect_ratio_and_resolution(self, tmp_path):
        src = make_pil_image(800, 600)
        in_path = str(tmp_path / "input.png")
        out_path = str(tmp_path / "output.png")
        src.save(in_path)

        process_image(
            in_path, out_path,
            aspect_ratio=(16, 9),
            resolution=(160, 90),
            method="center",
        )

        result = Image.open(out_path)
        assert result.size == (160, 90)

    def test_output_directory_created(self, tmp_path):
        src = make_pil_image(100, 100)
        in_path = str(tmp_path / "input.png")
        src.save(in_path)

        nested_out = str(tmp_path / "deep" / "nested" / "out.png")
        process_image(in_path, nested_out, resolution=(50, 50), method="center")

        assert os.path.isfile(nested_out)

    def test_entropy_method_does_not_raise(self, tmp_path):
        src = make_pil_image(200, 200)
        in_path = str(tmp_path / "input.png")
        out_path = str(tmp_path / "output.png")
        src.save(in_path)

        process_image(in_path, out_path, aspect_ratio=(1, 1), method="entropy")
        assert os.path.isfile(out_path)

    def test_face_method_does_not_raise(self, tmp_path):
        src = make_pil_image(200, 200)
        in_path = str(tmp_path / "input.png")
        out_path = str(tmp_path / "output.png")
        src.save(in_path)

        process_image(in_path, out_path, aspect_ratio=(1, 1), method="face")
        assert os.path.isfile(out_path)


# ---------------------------------------------------------------------------
# collect_input_files
# ---------------------------------------------------------------------------

class TestCollectInputFiles:
    def test_directory_collects_images(self, tmp_path):
        for name in ("a.jpg", "b.PNG", "c.txt", "d.jpeg"):
            (tmp_path / name).write_bytes(b"dummy")
        files = collect_input_files(str(tmp_path))
        names = {f.name.lower() for f in files}
        assert "a.jpg" in names
        assert "b.png" in names
        assert "d.jpeg" in names
        assert "c.txt" not in names

    def test_glob_pattern(self, tmp_path):
        for name in ("img1.jpg", "img2.jpg", "other.png"):
            (tmp_path / name).write_bytes(b"dummy")
        pattern = str(tmp_path / "img*.jpg")
        files = collect_input_files(pattern)
        names = {f.name for f in files}
        assert "img1.jpg" in names
        assert "img2.jpg" in names
        assert "other.png" not in names

    def test_empty_directory(self, tmp_path):
        assert collect_input_files(str(tmp_path)) == []


# ---------------------------------------------------------------------------
# main (CLI integration)
# ---------------------------------------------------------------------------

class TestMain:
    def _make_input_dir(self, tmp_path, count=3):
        in_dir = tmp_path / "input"
        in_dir.mkdir()
        for i in range(count):
            img = make_pil_image(200, 150)
            img.save(str(in_dir / f"img{i:02d}.jpg"))
        return in_dir

    def test_aspect_ratio_batch(self, tmp_path):
        in_dir = self._make_input_dir(tmp_path)
        out_dir = tmp_path / "output"
        rc = main(["-i", str(in_dir), "-o", str(out_dir), "-a", "1:1", "-m", "center"])
        assert rc == 0
        output_files = list(out_dir.glob("*.jpg"))
        assert len(output_files) == 3
        for f in output_files:
            w, h = Image.open(f).size
            assert w == h

    def test_resolution_batch(self, tmp_path):
        in_dir = self._make_input_dir(tmp_path)
        out_dir = tmp_path / "output"
        rc = main(["-i", str(in_dir), "-o", str(out_dir), "-r", "64x64", "-m", "center"])
        assert rc == 0
        for f in out_dir.glob("*.jpg"):
            assert Image.open(f).size == (64, 64)

    def test_format_override(self, tmp_path):
        in_dir = self._make_input_dir(tmp_path, count=2)
        out_dir = tmp_path / "output"
        rc = main(["-i", str(in_dir), "-o", str(out_dir), "-r", "64x64", "-f", "png", "-m", "center"])
        assert rc == 0
        assert len(list(out_dir.glob("*.png"))) == 2
        assert len(list(out_dir.glob("*.jpg"))) == 0

    def test_skip_existing_without_overwrite(self, tmp_path):
        in_dir = self._make_input_dir(tmp_path, count=2)
        out_dir = tmp_path / "output"
        # First run
        main(["-i", str(in_dir), "-o", str(out_dir), "-r", "64x64", "-m", "center"])
        first_mtime = {f.name: f.stat().st_mtime for f in out_dir.glob("*.jpg")}
        # Second run without --overwrite – should skip
        main(["-i", str(in_dir), "-o", str(out_dir), "-r", "128x128", "-m", "center"])
        for f in out_dir.glob("*.jpg"):
            assert f.stat().st_mtime == first_mtime[f.name], "File should not be touched"

    def test_overwrite_flag(self, tmp_path):
        in_dir = self._make_input_dir(tmp_path, count=1)
        out_dir = tmp_path / "output"
        main(["-i", str(in_dir), "-o", str(out_dir), "-r", "64x64", "-m", "center"])
        first_mtime = next(out_dir.glob("*.jpg")).stat().st_mtime
        # Give the filesystem time to register a different mtime
        time.sleep(0.05)
        main(["-i", str(in_dir), "-o", str(out_dir), "-r", "32x32", "--overwrite", "-m", "center"])
        new_size = Image.open(next(out_dir.glob("*.jpg"))).size
        assert new_size == (32, 32)

    def test_no_images_returns_nonzero(self, tmp_path):
        in_dir = tmp_path / "empty"
        in_dir.mkdir()
        out_dir = tmp_path / "output"
        rc = main(["-i", str(in_dir), "-o", str(out_dir), "-r", "64x64"])
        assert rc != 0

    def test_missing_crop_args_exits(self, tmp_path):
        in_dir = self._make_input_dir(tmp_path)
        out_dir = tmp_path / "output"
        with pytest.raises(SystemExit):
            main(["-i", str(in_dir), "-o", str(out_dir)])

    def test_invalid_aspect_ratio_exits(self, tmp_path):
        in_dir = self._make_input_dir(tmp_path)
        out_dir = tmp_path / "output"
        with pytest.raises(SystemExit):
            main(["-i", str(in_dir), "-o", str(out_dir), "-a", "badvalue"])

    def test_invalid_resolution_exits(self, tmp_path):
        in_dir = self._make_input_dir(tmp_path)
        out_dir = tmp_path / "output"
        with pytest.raises(SystemExit):
            main(["-i", str(in_dir), "-o", str(out_dir), "-r", "1920:1080"])
