import numpy as np
import sympy

from mathwiz.data import CLASSES, normalize_glyph
from mathwiz.recognize import decode, is_valid, segment, solve, to_expression


def test_implicit_multiplication():
    assert to_expression(list("3x+2(x-1)=4")) == "3*x+2*(x-1)=4"
    assert to_expression(list("(x+1)(x-1)")) == "(x+1)*(x-1)"


def test_solve():
    assert solve("12+30*2=") == 72
    assert solve("3*x+5=20") == [5]
    assert solve("x/4=8") == [32]
    assert solve("2+2=4") is sympy.true or solve("2+2=4") is True


def test_validity():
    assert is_valid(list("7x+1=239"))
    assert not is_valid(list("7x+(=239"))   # unbalanced
    assert not is_valid(list("3x+5"))       # x without an equation
    assert not is_valid(list("1==2"))


def test_decode_prefers_valid_reading():
    # glyph 3 is slightly more "(" than "1" by pixels, but only "1" makes sense
    probs = np.full((5, len(CLASSES)), 1e-4)
    for i, (best, p) in enumerate([("7", 0.9), ("+", 0.9), ("(", 0.55), ("=", 0.9), ("8", 0.9)]):
        probs[i, CLASSES.index(best)] = p
    probs[2, CLASSES.index("1")] = 0.4
    assert "".join(decode(probs)) == "7+1=8"


def test_segment_merges_stacked_parts_only():
    img = np.zeros((40, 60), np.float32)
    img[15:17, 2:20] = 1   # "=" top bar
    img[23:25, 2:20] = 1   # "=" bottom bar
    img[8:32, 30:33] = 1   # a "1" next to it
    glyphs, boxes = segment(img, min_pixels=5)
    assert len(glyphs) == 2
    assert boxes[0][2] == 2 and boxes[1][2] == 30


def test_normalize_glyph_centers_ink():
    g = normalize_glyph(np.pad(np.ones((10, 4)), 20))
    ys, xs = np.nonzero(g > 0.5)
    assert abs(ys.mean() - 14) < 1.5 and abs(xs.mean() - 14) < 1.5
