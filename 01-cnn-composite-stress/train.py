"""Train one model on Task A and report its relative L2 errors.

    python train.py unet            # or: mlp, unet_noskip
    python train.py unet --epochs 20
    python train.py unet --device cuda   # or mps on Apple silicon

Writes checkpoints/<name>.pt and results/<name>.json in this folder.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from models import MODELS, count_weights

HERE = Path(__file__).resolve().parent


def rel_l2(pred, y):
    """Mean over samples of ||pred - y||_2 / ||y||_2: layouts with high and
    low stress count equally."""
    return ((pred - y).flatten(1).norm(dim=1) / y.flatten(1).norm(dim=1)).mean()


@torch.no_grad()
def error(model, X, Y, bs=200):
    model.eval()
    e = sum(rel_l2(model(X[i:i + bs]), Y[i:i + bs]).item() * len(X[i:i + bs])
            for i in range(0, len(X), bs)) / len(X)
    model.train()
    return e


def train(model, X, Y, epochs=100, bs=32, log=None):
    """The same loop for every model: Adam, learning rate 1e-3 decreasing to 0
    along a cosine over the epochs, mini-batches of 32."""
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        for idx in torch.randperm(len(X), device=X.device).split(bs):
            loss = rel_l2(model(X[idx]), Y[idx])
            opt.zero_grad()
            loss.backward()                        # backpropagation
            opt.step()                             # update the weights
        sched.step()
        if log:
            log(ep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=MODELS)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    d = np.load(HERE / "data" / "taskA.npz")
    Xtr, Ytr, Xte, Yte = (torch.from_numpy(d[k]).to(args.device) for k in
                          ("x_train", "y_train", "x_test", "y_test"))

    model = MODELS[args.model]().to(args.device)
    print(f"{args.model}: {count_weights(model):,} weights, "
          f"{len(Xtr)} training and {len(Xte)} test layouts")

    history = {"train": [], "test": []}

    def log(ep):
        history["train"].append(error(model, Xtr, Ytr))
        history["test"].append(error(model, Xte, Yte))
        if (ep + 1) % 10 == 0:
            print(f"epoch {ep + 1:3d}  train {history['train'][-1]:.2%}  "
                  f"test {history['test'][-1]:.2%}")

    t = time.perf_counter()
    train(model, Xtr, Ytr, epochs=args.epochs, log=log)
    t = time.perf_counter() - t

    res = {"model": args.model, "weights": count_weights(model),
           "epochs": args.epochs, "seed": args.seed, "device": args.device,
           "train_error": history["train"][-1], "test_error": history["test"][-1],
           "train_seconds": round(t, 1), "history": history}
    (HERE / "checkpoints").mkdir(exist_ok=True)
    (HERE / "results").mkdir(exist_ok=True)
    torch.save(model.cpu().state_dict(), HERE / "checkpoints" / f"{args.model}.pt")
    (HERE / "results" / f"{args.model}.json").write_text(json.dumps(res, indent=1))
    print(f"done in {t:.0f} s: train error {res['train_error']:.1%}, "
          f"test error {res['test_error']:.1%}")


if __name__ == "__main__":
    main()
