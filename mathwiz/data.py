"""Training data: MNIST digits plus synthesized handwritten operator symbols."""
import gzip
import math
import random
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

SYMBOLS = ["+", "-", "*", "/", "=", "(", ")", "x"]
CLASSES = [str(d) for d in range(10)] + SYMBOLS


def _download(name):
    path = DATA_DIR / name
    if not path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        urllib.request.urlretrieve(MNIST_MIRROR + name, path)
    return path


def load_mnist(split):
    with gzip.open(_download(MNIST_FILES[f"{split}_images"])) as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    with gzip.open(_download(MNIST_FILES[f"{split}_labels"])) as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    return images, labels.astype(np.int64)


def normalize_glyph(mask):
    """MNIST-style normalization: fit the ink into 20x20 keeping aspect ratio,
    then place it in a 28x28 canvas centred on its centre of mass."""
    ys, xs = np.nonzero(mask > 0.1)
    if len(ys) == 0:
        return np.zeros((28, 28), np.float32)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
    h, w = crop.shape
    scale = 20.0 / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    img = Image.fromarray((crop * 255).astype(np.uint8)).resize((nw, nh), Image.LANCZOS)
    canvas = np.zeros((28, 28), np.float32)
    top, left = (28 - nh) // 2, (28 - nw) // 2
    canvas[top:top + nh, left:left + nw] = np.asarray(img, np.float32) / 255.0
    cy, cx = ndimage.center_of_mass(canvas)
    if not math.isnan(cy):
        canvas = ndimage.shift(canvas, (14 - cy, 14 - cx), order=1, mode="constant")
    return np.clip(canvas, 0, 1)


def _stroke(draw, points, width):
    draw.line(points, fill=255, width=width, joint="curve")
    r = width / 2
    for x, y in (points[0], points[-1]):
        draw.ellipse((x - r, y - r, x + r, y + r), fill=255)


def _curve(cx, cy, rx, ry, a0, a1, n=12):
    return [(cx + rx * math.cos(a0 + (a1 - a0) * t / n), cy + ry * math.sin(a0 + (a1 - a0) * t / n)) for t in range(n + 1)]


def draw_symbol(symbol, rng):
    """Render one operator as if drawn with a pen: jittered strokes, varying
    thickness, slant and proportions. Returns a 28x28 float image."""
    size = 96
    img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)
    j = lambda v: v + rng.uniform(-5, 5)  # noqa: E731  (hand jitter)
    w = rng.randint(6, 12)
    c = size / 2
    if symbol == "+":
        arm = rng.uniform(22, 34)
        _stroke(d, [(j(c - arm), j(c)), (j(c + arm), j(c))], w)
        _stroke(d, [(j(c), j(c - arm)), (j(c), j(c + arm))], w)
    elif symbol == "-":
        arm = rng.uniform(22, 36)
        y = j(c)
        _stroke(d, [(j(c - arm), y), (j(c), y + rng.uniform(-3, 3)), (j(c + arm), y + rng.uniform(-4, 4))], w)
    elif symbol == "*":  # multiplication cross
        arm = rng.uniform(16, 26)
        _stroke(d, [(j(c - arm), j(c - arm)), (j(c + arm), j(c + arm))], w)
        _stroke(d, [(j(c + arm), j(c - arm)), (j(c - arm), j(c + arm))], w)
    elif symbol == "/":  # division, handwritten as an obelus: a bar with a dot above and below
        arm = rng.uniform(20, 32)
        gap = rng.uniform(13, 20)
        _stroke(d, [(j(c - arm), j(c)), (j(c + arm), j(c))], w)
        for y in (c - gap, c + gap):
            r = w * rng.uniform(0.6, 0.9)
            x0, y0 = j(c), j(y)
            d.ellipse((x0 - r, y0 - r, x0 + r, y0 + r), fill=255)
    elif symbol == "=":
        arm = rng.uniform(22, 34)
        gap = rng.uniform(9, 16)
        for y in (c - gap, c + gap):
            _stroke(d, [(j(c - arm), j(y)), (j(c + arm), j(y))], w)
    elif symbol in "()":
        h = rng.uniform(32, 42)
        bow = rng.uniform(12, 22)
        # left half of an ellipse is "("; mirror it for ")"
        pts = _curve(c + bow, c, bow, h, math.pi / 2 + 0.4, 3 * math.pi / 2 - 0.4)
        if symbol == ")":
            pts = [(2 * c - x, y) for x, y in pts]
        _stroke(d, pts, w)
    elif symbol == "x":  # the variable: two curves back to back, unlike the straight multiplication cross
        r = rng.uniform(14, 20)
        _stroke(d, _curve(c - r * 0.9, c, r, r * 1.2, -1.2, 1.2), w)
        _stroke(d, _curve(c + r * 0.9, c, r, r * 1.2, math.pi - 1.2, math.pi + 1.2), w)
    img = img.rotate(rng.uniform(-14, 14), resample=Image.BILINEAR)
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.5)))
    arr = np.asarray(img, np.float32) / 255.0
    return normalize_glyph(arr)


def synth_symbols(n_per_class, seed):
    rng = random.Random(seed)
    images, labels = [], []
    for k, sym in enumerate(SYMBOLS):
        for _ in range(n_per_class):
            images.append(draw_symbol(sym, rng))
            labels.append(10 + k)
    return np.stack(images), np.array(labels, np.int64)
