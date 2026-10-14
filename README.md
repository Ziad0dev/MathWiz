# MathWiz
Handwritten equation solver using CNN

```
image ─► segment into glyphs ─► CNN per glyph ─► syntax-aware decoding ─► SymPy ─► answer
```

- **Segmentation:** connected components, with vertically stacked pieces merged (the bars of `=`, the dots of `÷`, a detached hat of `5`); each glyph is normalized MNIST-style (20×20 box, centred on its centre of mass)
- **CNN:** a small 4-conv network over 18 classes: digits `0-9` and `+ − × ÷ = ( ) x`. Digits come from MNIST; operators are synthesized as jittered pen strokes (no public operator dataset without an account), with `÷` drawn as an obelus because a handwritten `/` is indistinguishable from a `1`
- **Decoding:** beam search over each glyph's top-3 classes, keeping the most probable reading that is a valid expression or equation (balanced brackets, one `=`, ...)
- **Solving:** SymPy evaluates `12+30×2=` or solves for `x` in `3x+5=20`, with implicit multiplication (`2(x+1)`)

```sh
nix-shell                      # or: pip install -r requirements.txt
python -m mathwiz train        # ~3 min on CPU, downloads MNIST
python -m mathwiz evaluate     # end-to-end accuracy on unseen handwriting
python -m mathwiz demo         # draws a random equation to equation.png and solves it
python -m mathwiz solve photo.png
python -m pytest tests
```

## Results
| | |
| --- | --- |
| held-out glyph accuracy (MNIST test digits) | 99.6 % |
| equations read exactly, per-glyph argmax | 63 % |
| equations read exactly, with syntax-aware decoding | 79 % |

Equations are composed from MNIST **test** digits and operator strokes from an unseen random seed. One wrong glyph out of ~8 ruins an equation, which is why exact-read accuracy is much lower than glyph accuracy. Remaining errors are real digit confusions (8↔9, 6↔0). Operators are synthetic, so real photos of paper will be harder; collecting a few hundred real operator samples is the obvious next step.
