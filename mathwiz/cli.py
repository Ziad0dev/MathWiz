"""Command line: train the CNN, evaluate end to end, solve an image.

    python -m mathwiz train
    python -m mathwiz evaluate              # end-to-end accuracy on synthetic equations
    python -m mathwiz demo -o equation.png  # render a random equation and solve it
    python -m mathwiz solve photo.png
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import CLASSES, load_mnist, synth_symbols, draw_symbol
from .model import GlyphNet
from .recognize import decode, glyph_probs, load_image, load_model, segment, solve, to_expression

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "glyphnet.pt"


def cmd_train(args):
    torch.manual_seed(0)
    digits, digit_labels = load_mnist("train")
    digits = digits.astype(np.float32) / 255.0
    t = time.time()
    syms, sym_labels = synth_symbols(args.per_symbol, seed=1)
    print(f"synthesized {len(syms)} operator glyphs in {time.time() - t:.0f}s")
    x = torch.from_numpy(np.concatenate([digits, syms])).unsqueeze(1)
    y = torch.from_numpy(np.concatenate([digit_labels, sym_labels]))
    loader = DataLoader(TensorDataset(x, y), batch_size=128, shuffle=True)

    model = GlyphNet()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=args.epochs * len(loader))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    for epoch in range(args.epochs):
        model.train()
        total, correct, t = 0, 0, time.time()
        for xb, yb in loader:
            # random shifts make the model robust to imperfect segmentation crops
            xb = torch.roll(xb, shifts=(random.randint(-2, 2), random.randint(-2, 2)), dims=(2, 3))
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            sched.step()
            correct += (out.argmax(1) == yb).sum().item()
            total += len(yb)
        print(f"epoch {epoch + 1}: train accuracy {correct / total:.4f} ({time.time() - t:.0f}s)")

    # held-out glyph accuracy: MNIST test digits + symbols from an unseen seed
    test_digits, test_labels = load_mnist("test")
    test_syms, test_sym_labels = synth_symbols(300, seed=99)
    tx = torch.from_numpy(np.concatenate([test_digits.astype(np.float32) / 255.0, test_syms])).unsqueeze(1)
    ty = torch.from_numpy(np.concatenate([test_labels, test_sym_labels]))
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(b).argmax(1) for b in tx.split(1000)])
    print(f"held-out glyph accuracy: {(pred == ty).float().mean():.4f}")
    for k, name in enumerate(CLASSES[10:], start=10):
        mask = ty == k
        print(f"  {name}: {(pred[mask] == k).float().mean():.3f}", end="")
    print()
    MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"saved {MODEL_PATH}")


def random_equation(rng):
    a, b, c = rng.randint(1, 9), rng.randint(2, 9), rng.randint(10, 99)
    templates = [
        f"{c}+{a}*{b}=",
        f"({a}+{b})*{a}=",
        f"{a*b}/{b}-{a}=",
        f"{b}x+{a}={b * c + a}",
        f"{a}x-{b}={a * c - b}" if a * c - b > 0 else f"{a}x={a * c}",
        f"x/{b}={c}",
        f"{c}-x={a}",
        f"{a}(x+{b})={a * (c + b)}",
    ]
    return rng.choice(templates)


def render(expression, rng, digit_pool, height=40):
    """Lay out handwritten glyphs for `expression` on a white canvas, like ink on paper."""
    glyphs = []
    for ch in expression:
        if ch.isdigit():
            choices = digit_pool[int(ch)]
            img = choices[rng.randrange(len(choices))].astype(np.float32) / 255.0
        else:
            img = draw_symbol(ch, rng)
        # trim to the ink and scale to a handwriting-like size (operators are smaller)
        ys, xs = np.nonzero(img > 0.1)
        img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        scale = height / 28 * (1.0 if ch.isdigit() or ch in "()x" else 0.7) * rng.uniform(0.85, 1.1)
        h, w = max(2, int(img.shape[0] * scale)), max(2, int(img.shape[1] * scale))
        glyphs.append(np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR), np.float32) / 255.0)
    width = sum(g.shape[1] for g in glyphs) + 14 * (len(glyphs) + 2)
    canvas = np.zeros((height * 2, width), np.float32)
    x = 14
    for g in glyphs:
        y = (canvas.shape[0] - g.shape[0]) // 2 + rng.randint(-3, 3)
        region = canvas[y:y + g.shape[0], x:x + g.shape[1]]
        np.maximum(region, g, out=region)
        x += g.shape[1] + rng.randint(9, 16)
    return Image.fromarray(((1 - canvas) * 255).astype(np.uint8))  # dark ink on white


def recognize_file(model, path, use_syntax=True):
    glyphs, _ = segment(load_image(path))
    probs = glyph_probs(model, glyphs)
    symbols = decode(probs) if use_syntax else [CLASSES[i] for i in probs.argmax(1)]
    return to_expression(symbols), probs.max(1).tolist() if len(probs) else []


def digit_pool(split="test"):
    images, labels = load_mnist(split)
    return {d: images[labels == d] for d in range(10)}


def cmd_demo(args):
    rng = random.Random(args.seed)
    equation = random_equation(rng)
    render(equation, rng, digit_pool()).save(args.out)
    print(f"wrote {args.out} (drew: {equation})")
    args.image = args.out
    cmd_solve(args)


def cmd_solve(args):
    model = load_model(MODEL_PATH)
    expression, conf = recognize_file(model, args.image)
    print(f"read:   {expression}   (lowest glyph confidence {min(conf, default=0):.2f})")
    try:
        print(f"answer: {solve(expression)}")
    except Exception as e:  # unbalanced brackets, misread glyphs, ...
        print(f"could not solve '{expression}': {e}")


def cmd_evaluate(args):
    import tempfile

    model = load_model(MODEL_PATH)
    rng = random.Random(1234)
    pool = digit_pool()
    exact = solved = exact_raw = 0
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(args.n):
            equation = random_equation(rng)
            path = Path(tmp) / f"{i}.png"
            render(equation, rng, pool).save(path)
            read, _ = recognize_file(model, path)
            exact_raw += recognize_file(model, path, use_syntax=False)[0] == to_expression(list(equation))
            truth = to_expression(list(equation))
            ok = read == truth
            exact += ok
            try:
                solved += str(solve(read)) == str(solve(truth))
            except Exception:
                pass
            if not ok and len(failures) < 8:
                failures.append(f"{truth}  read as  {read}")
    print(f"{args.n} handwritten equations (MNIST test digits + unseen operator strokes)")
    print(f"  read exactly, per-glyph argmax: {exact_raw / args.n:.1%}")
    print(f"  read exactly, syntax-aware decoding: {exact / args.n:.1%}")
    print(f"  correct answer: {solved / args.n:.1%}")
    if failures:
        print("  examples of misreads:\n    " + "\n    ".join(failures))


def main():
    parser = argparse.ArgumentParser(prog="mathwiz", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--per-symbol", type=int, default=3000)
    p = sub.add_parser("evaluate")
    p.add_argument("-n", type=int, default=200)
    p = sub.add_parser("demo")
    p.add_argument("-o", "--out", default="equation.png")
    p.add_argument("--seed", type=int, default=7)
    p = sub.add_parser("solve")
    p.add_argument("image")
    args = parser.parse_args()
    {"train": cmd_train, "evaluate": cmd_evaluate, "demo": cmd_demo, "solve": cmd_solve}[args.cmd](args)
