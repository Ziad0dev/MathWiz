"""Segment an equation image into glyphs, classify them, and solve."""
import numpy as np
import sympy
import torch
from PIL import Image
from scipy import ndimage

from .data import CLASSES, normalize_glyph
from .model import GlyphNet


def load_image(path):
    """Grayscale in [0, 1] with ink = 1, whatever the input polarity."""
    img = np.asarray(Image.open(path).convert("L"), np.float32) / 255.0
    if np.median(img) > 0.5:  # dark ink on light paper
        img = 1.0 - img
    # Drop paper texture but keep the soft anti-aliased stroke edges the CNN was
    # trained on (MNIST strokes are not binary); a high threshold thins every glyph.
    return np.where(img > 0.2, img, 0.0)


def segment(ink, min_pixels=12):
    """Connected components, merged when they are stacked vertically
    (the two bars of "=" or the dots and bar of a division sign)."""
    labels, n = ndimage.label(ink > 0.1)
    boxes = []
    for i, sl in enumerate(ndimage.find_objects(labels), 1):
        if sl is None or (labels[sl] == i).sum() < min_pixels:
            continue
        boxes.append([sl[0].start, sl[0].stop, sl[1].start, sl[1].stop, [i]])
    boxes.sort(key=lambda b: b[2])
    merged = []
    for b in boxes:
        if merged:
            p = merged[-1]
            overlap = min(p[3], b[3]) - max(p[2], b[2])
            # stacked parts (the bars of "=", the dots of "÷", a detached stroke of "5")
            # share columns but not rows; a neighbour beside a glyph shares rows
            v_overlap = min(p[1], b[1]) - max(p[0], b[0])
            if overlap > 0.6 * min(p[3] - p[2], b[3] - b[2]) and v_overlap < 0.25 * min(p[1] - p[0], b[1] - b[0]):
                p[0], p[1], p[2], p[3] = min(p[0], b[0]), max(p[1], b[1]), min(p[2], b[2]), max(p[3], b[3])
                p[4] += b[4]
                continue
        merged.append(b)
    glyphs = []
    for y0, y1, x0, x1, ids in merged:
        mask = np.isin(labels[y0:y1, x0:x1], ids) * ink[y0:y1, x0:x1]
        glyphs.append(normalize_glyph(mask))
    return glyphs, [(b[0], b[1], b[2], b[3]) for b in merged]


def load_model(path):
    model = GlyphNet()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()


@torch.no_grad()
def glyph_probs(model, glyphs):
    if not glyphs:
        return np.zeros((0, len(CLASSES)))
    x = torch.from_numpy(np.stack(glyphs)).unsqueeze(1)
    return torch.softmax(model(x), dim=1).numpy()


def classify(model, glyphs):
    """Most likely class per glyph, ignoring context."""
    probs = glyph_probs(model, glyphs)
    return [CLASSES[i] for i in probs.argmax(1)], probs.max(1).tolist() if len(probs) else []


def is_valid(symbols):
    """Could this sequence be a well-formed expression or equation?"""
    text = "".join(symbols)
    if not text or text.count("=") > 1 or ("x" in text and "=" not in text):
        return False
    depth = 0
    for ch in text:
        depth += {"(": 1, ")": -1}.get(ch, 0)
        if depth < 0:
            return False
    if depth:
        return False
    if text[0] in "*/=)" or text[-1] in "+-*/(":
        return False
    try:
        expr = to_expression(symbols)
        from sympy.parsing.sympy_parser import parse_expr
        for side in expr.split("="):
            if side:
                parse_expr(side, local_dict={"x": X}, evaluate=False)
        return True
    except Exception:
        return False


def decode(probs, beam=64, top_k=3):
    """Beam search over each glyph's top-k classes; return the most probable
    reading that is syntactically valid (fall back to the plain argmax)."""
    beams = [(0.0, [])]
    for row in probs:
        options = np.argsort(row)[::-1][:top_k]
        beams = sorted(
            ((score + float(np.log(row[c] + 1e-9)), seq + [CLASSES[c]]) for score, seq in beams for c in options),
            key=lambda t: -t[0],
        )[:beam]
    for _score, seq in beams:
        if is_valid(seq):
            return seq
    return [CLASSES[i] for i in probs.argmax(1)]


X = sympy.Symbol("x")


def to_expression(symbols):
    """Join glyphs into a SymPy-parsable string, adding implicit multiplication (2x, 3(…), x(…))."""
    out = []
    for s in symbols:
        if out and ((out[-1].isdigit() or out[-1] in "x)") and (s == "x" or s == "(")
                    or (out[-1] in "x)" and s.isdigit())):
            out.append("*")
        out.append(s)
    return "".join(out)


def solve(expression):
    """Evaluate an arithmetic expression or solve an equation for x."""
    from sympy.parsing.sympy_parser import parse_expr

    if "=" in expression:
        left, right = expression.split("=", 1)
        if not left:
            raise ValueError("nothing left of '='")
        lhs = parse_expr(left, local_dict={"x": X})
        if not right:  # "12+30=" means "what is 12+30"
            return sympy.nsimplify(lhs)
        rhs = parse_expr(right, local_dict={"x": X})
        if X in (lhs - rhs).free_symbols:
            return sympy.solve(sympy.Eq(lhs, rhs), X)
        return sympy.simplify(lhs - rhs) == 0  # check an equality like 2+2=4
    return sympy.nsimplify(parse_expr(expression, local_dict={"x": X}))
