"""Figures for Task A, written to figures/.

    python figures.py

fig_data.png      three test layouts and their FEM stress maps
fig_pred.png      one test layout: FEM and the predictions of the trained models
fig_curves.png    training and test error per epoch
Models that have not been trained yet are skipped.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models import MODELS
from train import rel_l2

LABELS = {"mlp": "MLP", "unet": "U-Net", "unet_noskip": "U-Net, no skips"}
OUT = Path("figures")


def show(ax, img, title, **kw):
    im = ax.imshow(img, origin="lower", extent=(0, 1, 0, 1), **kw)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def fig_data(X, Y, idx=(0, 1, 2)):
    fig, axs = plt.subplots(2, len(idx), figsize=(3 * len(idx), 6), layout="constrained")
    vmax = max(Y[i].max() for i in idx)
    for c, i in enumerate(idx):
        show(axs[0, c], X[i], f"input x: log10 E  (test #{i})", cmap="Greys", vmin=0, vmax=1)
        im = show(axs[1, c], Y[i], "output y: s (FEM)", cmap="viridis", vmin=0, vmax=vmax)
    fig.colorbar(im, ax=axs[1, :], shrink=0.8, label="s = σ_vm / (E_m ε̄)")
    fig.savefig(OUT / "fig_data.png", dpi=200)
    plt.close(fig)


def fig_pred(X, Y, models, i=0):
    x, y = torch.from_numpy(X[i:i + 1]), torch.from_numpy(Y[i:i + 1])
    panels = [("input x: log10 E", X[i], dict(cmap="Greys", vmin=0, vmax=1)),
              ("FEM", Y[i], None)]
    with torch.no_grad():
        for name, model in models.items():
            p = model(x)
            panels.append((f"{LABELS[name]}  ({rel_l2(p, y).item():.1%})", p[0].numpy(), None))
    vmax = Y[i].max()
    fig, axs = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.4), layout="constrained")
    for ax, (title, img, kw) in zip(axs, panels):
        im = show(ax, img, title, **(kw or dict(cmap="viridis", vmin=0, vmax=vmax)))
    fig.colorbar(im, ax=axs[1:], shrink=0.8, label="s")
    fig.savefig(OUT / "fig_pred.png", dpi=200)
    plt.close(fig)


def fig_curves(results):
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    for k, (name, r) in enumerate(results.items()):
        ep = np.arange(1, len(r["history"]["test"]) + 1)
        ax.plot(ep, r["history"]["train"], "--", color=f"C{k}", label=f"{LABELS[name]}, training")
        ax.plot(ep, r["history"]["test"], "-", color=f"C{k}", label=f"{LABELS[name]}, test")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean relative L2 error")
    ax.legend(fontsize=8)
    fig.savefig(OUT / "fig_curves.png", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    d = np.load("data/taskA.npz")
    X, Y = d["x_test"], d["y_test"]
    fig_data(X, Y)

    models, results = {}, {}
    for name in MODELS:
        if Path(f"checkpoints/{name}.pt").exists():
            m = MODELS[name]()
            m.load_state_dict(torch.load(f"checkpoints/{name}.pt", map_location="cpu"))
            models[name] = m.eval()
            results[name] = json.loads(Path(f"results/{name}.json").read_text())
    if models:
        fig_pred(X, Y, models)
        fig_curves(results)
        for name, r in results.items():
            print(f"{LABELS[name]:16s} {r['weights']:>9,} weights   "
                  f"train {r['train_error']:.1%}   test {r['test_error']:.1%}")
    print(f"figures written to {OUT}/")


if __name__ == "__main__":
    main()
