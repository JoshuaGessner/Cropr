# Cropr
Intelligent batch cropping of image sets

Cropr crops sets of images to a user-specified aspect ratio and/or resolution, using subject detection to keep the most important content in frame.

## Requirements

- Python 3.8+
- [Pillow](https://python-pillow.org/) >= 9.0.0
- [OpenCV](https://opencv.org/) >= 4.5.0
- [NumPy](https://numpy.org/) >= 1.21.0

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

```
python cropr.py -i PATH -o DIR [options]
```

### Required Arguments

| Argument | Description |
|---|---|
| `-i PATH`, `--input PATH` | Input directory or glob pattern (e.g. `./photos` or `"./raw/**/*.jpg"`) |
| `-o DIR`, `--output DIR` | Output directory (created automatically if it does not exist) |

At least one of `--aspect-ratio` or `--resolution` must also be provided.

### Options

| Argument | Description |
|---|---|
| `-a W:H`, `--aspect-ratio W:H` | Target aspect ratio, e.g. `16:9`, `4:3`, `1:1` |
| `-r WxH`, `--resolution WxH` | Target output resolution in pixels, e.g. `1920x1080` or `512x512` |
| `-m METHOD`, `--method METHOD` | Subject detection method: `face` (default), `entropy`, or `center` |
| `-f EXT`, `--format EXT` | Force output image format, e.g. `jpg` or `png` (preserves source format when omitted) |
| `--overwrite` | Overwrite existing output files (skipped by default) |

### Subject Detection Methods

- **`face`** *(default)* – Uses Haar-cascade face and body detection. Falls back to entropy-based detection when no face or body is found.
- **`entropy`** – Uses gradient-based saliency to find the most visually interesting region.
- **`center`** – Always crops from the center of the image, with no detection.

## Examples

```bash
# Crop all images in ./photos to a 16:9 aspect ratio (face-aware)
python cropr.py -i ./photos -o ./output -a 16:9

# Crop and resize to exactly 512×512 pixels
python cropr.py -i ./photos -o ./output -r 512x512

# Combine aspect-ratio crop with a target resolution
python cropr.py -i ./photos -o ./output -a 1:1 -r 512x512

# Use simple center-crop (no subject detection)
python cropr.py -i ./photos -o ./output -a 4:3 --method center

# Process a glob pattern and convert all output files to PNG
python cropr.py -i "./raw/**/*.jpg" -o ./output -a 16:9 --format png

# Overwrite any existing files in the output directory
python cropr.py -i ./photos -o ./output -a 16:9 --overwrite
```
