"""Figures for Task A, written to figures/ in this folder (sized for slides).

    python figures.py

fig_example.png   one test layout and its FEM stress map
fig_data.png      three test layouts and their FEM stress maps
fig_pred.png      one test layout: FEM and the predictions of the trained models
fig_curves.png    training and test error per epoch
map_input.png, map_stress.png   the bare maps of one layout (for diagrams)
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

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "savefig.dpi": 300})
LABELS = {"mlp": "MLP", "unet": "U-Net", "unet_noskip": "U-Net, no skips"}
INPUT = dict(cmap="Greys", vmin=0, vmax=1)
HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"


def show(ax, img, title, **kw):
    im = ax.imshow(img, origin="lower", extent=(0, 1, 0, 1), **kw)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def fig_example(X, Y, i=0):
    fig, axs = plt.subplots(1, 2, figsize=(3.6, 1.75), layout="constrained")
    show(axs[0], X[i], "input x: fibre layout", **INPUT)
    im = show(axs[1], Y[i], "output y: stress s (FEM)", cmap="viridis", vmin=0)
    fig.colorbar(im, ax=axs[1], shrink=0.9)
    fig.savefig(OUT / "fig_example.png")
    plt.close(fig)


def fig_maps(X, Y, i=0):
    """Input and FEM stress map of one layout as bare images, no axes."""
    up = np.ones((8, 8))                       # 32 x 32 -> 256 x 256, sharp cells
    plt.imsave(OUT / "map_input.png", np.kron(X[i], up)[::-1], cmap="Greys", vmin=0, vmax=1)
    plt.imsave(OUT / "map_stress.png", np.kron(Y[i], up)[::-1], cmap="viridis", vmin=0)


def fig_data(X, Y, idx=(0, 1, 2)):
    fig, axs = plt.subplots(2, len(idx), figsize=(4.2, 2.9), layout="constrained")
    vmax = max(Y[i].max() for i in idx)
    for c, i in enumerate(idx):
        show(axs[0, c], X[i], f"x, test layout {i}", **INPUT)
        im = show(axs[1, c], Y[i], f"y = s (FEM), layout {i}", cmap="viridis", vmin=0, vmax=vmax)
    fig.colorbar(im, ax=axs[1, :], shrink=0.9, label="s")
    fig.savefig(OUT / "fig_data.png")
    plt.close(fig)


def fig_pred(X, Y, models, i=0):
    x, y = torch.from_numpy(X[i:i + 1]), torch.from_numpy(Y[i:i + 1])
    panels = [("input x", X[i], INPUT), ("FEM", Y[i], None)]
    with torch.no_grad():
        for name, model in models.items():
            p = model(x)
            panels.append((f"{LABELS[name]}: {rel_l2(p, y).item():.1%}", p[0].numpy(), None))
    vmax = Y[i].max()
    fig, axs = plt.subplots(1, len(panels), figsize=(1.15 * len(panels) + 0.4, 1.45),
                            layout="constrained")
    for ax, (title, img, kw) in zip(axs, panels):
        im = show(ax, img, title, **(kw or dict(cmap="viridis", vmin=0, vmax=vmax)))
    fig.colorbar(im, ax=axs[1:], shrink=0.9, label="s")
    fig.savefig(OUT / "fig_pred.png")
    plt.close(fig)


def fig_curves(results):
    fig, ax = plt.subplots(figsize=(3.4, 2.4), layout="constrained")
    for k, (name, r) in enumerate(results.items()):
        ep = np.arange(1, len(r["history"]["test"]) + 1)
        ax.plot(ep, r["history"]["train"], "--", color=f"C{k}", lw=1)
        ax.plot(ep, r["history"]["test"], "-", color=f"C{k}", lw=1.2, label=LABELS[name])
    ax.plot([], [], "k-", lw=1.2, label="test")
    ax.plot([], [], "k--", lw=1, label="training")
    ax.set_yscale("log")
    ax.set_yticks([0.04, 0.06, 0.1, 0.2, 0.4], ["4%", "6%", "10%", "20%", "40%"])
    ax.minorticks_off()
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean relative L2 error")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.savefig(OUT / "fig_curves.png")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    d = np.load(HERE / "data" / "taskA.npz")
    X, Y = d["x_test"], d["y_test"]
    fig_example(X, Y)
    fig_maps(X, Y)
    fig_data(X, Y)

    models, results = {}, {}
    for name in MODELS:
        ckpt = HERE / "checkpoints" / f"{name}.pt"
        if ckpt.exists():
            m = MODELS[name]()
            m.load_state_dict(torch.load(ckpt, map_location="cpu"))
            models[name] = m.eval()
            results[name] = json.loads((HERE / "results" / f"{name}.json").read_text())
    if models:
        fig_pred(X, Y, models)
        fig_curves(results)
        for name, r in results.items():
            print(f"{LABELS[name]:16s} {r['weights']:>9,} weights   "
                  f"train {r['train_error']:.1%}   test {r['test_error']:.1%}")
    print(f"figures written to {OUT}/")


if __name__ == "__main__":
    main()
