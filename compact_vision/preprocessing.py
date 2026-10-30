"""One RGB/resize/crop/normalization contract for training, export and serving."""

import io
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

CONTRACT = {
    "color": "RGB", "exif": "transpose before RGB conversion", "resize_short_side": 256,
    "crop": 224, "interpolation": "PIL bilinear", "layout": "NCHW", "dtype": "float32",
    "scale": 255.0, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
}


def decode_image(encoded, *, max_bytes=8_388_608, max_pixels=12_000_000):
    if not encoded or len(encoded) > max_bytes:
        raise ValueError("image is empty or exceeds the encoded-byte limit")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(encoded)) as image:
                if image.format not in ("JPEG", "PNG"):
                    raise ValueError("only JPEG and PNG are supported")
                if image.width * image.height > max_pixels or getattr(image, "n_frames", 1) != 1:
                    raise ValueError("image exceeds pixel limit or contains multiple frames")
                image.load()  # Materialize lazy decoding inside the timed decode boundary.
                return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ValueError("invalid or unsafe image encoding") from error


def preprocess(image):
    image = image.convert("RGB")
    width, height = image.size
    short = CONTRACT["resize_short_side"]
    size = (int(short * width / height), short) if width >= height else (short, int(short * height / width))
    image = image.resize(size, resample=Image.Resampling.BILINEAR)
    crop = CONTRACT["crop"]
    left, top = int(round((image.width - crop) / 2)), int(round((image.height - crop) / 2))
    pixels = np.asarray(image.crop((left, top, left + crop, top + crop)), dtype=np.float32)
    values = pixels / np.float32(CONTRACT["scale"])
    values = (values - np.array(CONTRACT["mean"], dtype=np.float32)) / np.array(CONTRACT["std"], dtype=np.float32)
    return np.ascontiguousarray(values.transpose(2, 0, 1)[None])


def describe(values):
    return {
        "shape": list(values.shape), "dtype": str(values.dtype),
        "min": float(values.min()), "max": float(values.max()),
        "first_values": values.reshape(-1)[:8].tolist(),
    }
