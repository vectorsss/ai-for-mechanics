# CNN for stress fields in fibre composites

Code for the mechanics example in **Lecture 1: Convolutional Neural Networks**.
Given the fibre layout of a composite plate, a network predicts the stress map
that a finite element (FEM) solver would compute. Everything runs on a laptop
CPU in a few minutes; no GPU and no downloaded data are needed.

## The problem

- Plate: the unit square. Soft matrix with Young's modulus $E_m = 1$, stiff
  circular fibres with $E_f = 10$, Poisson's ratio $\nu = 0.3$, linear
  elasticity, plane stress.
- Loading: left edge $u_x = 0$, right edge $u_x = \bar\varepsilon = 0.01$
  (1 % of the width), bottom-left corner $u_y = 0$; top and bottom are free.
- Fibres: $N_f \sim \mathcal U\{3,\dots,8\}$ discs, centres
  $\sim \mathcal U(0,1)^2$, radii $\sim \mathcal U(0.06, 0.15)$. They may
  overlap or be cut by the edge.
- Mesh: $32 \times 32$ bilinear (Q1) elements. An element is fibre if its
  centre lies inside a fibre.

| | meaning | shape |
|---|---|---|
| input $x$ | $\log_{10} E$ in each element: 0 in the matrix, 1 in a fibre | 32 × 32 |
| output $y$ | $s = \sigma_{vm} / (E_m \bar\varepsilon)$ at each element centre | 32 × 32 |

$\sigma_{vm}$ is the von Mises stress; $s = 1$ is the stress of a plate
without fibres. 1000 random layouts: 800 for training, 200 for testing.

## Run

```sh
pip install -r requirements.txt   # numpy, scipy, matplotlib, torch

python fem.py                     # 1000 layouts + FEM solves -> data/taskA.npz
python train.py mlp               # fully connected baseline
python train.py unet              # the CNN (U-Net)
python train.py unet_noskip       # the same U-Net without its skip connections
python figures.py                 # figures/*.png and a summary table
```

`train.py --epochs 20` gives a quicker, less accurate run; `--device cuda` or
`--device mps` trains on a GPU.

## Expected results

Mean relative $L^2$ error, $\frac{1}{N}\sum_i \lVert f_\theta(x_i) - y_i\rVert_2 / \lVert y_i \rVert_2$,
measured against FEM on the same 32 × 32 mesh. 100 epochs, seed 0.

| model | weights | training error | test error | training time (CPU) |
|---|---:|---:|---:|---:|
| MLP | 1,312,768 | 9.8 % | 22.6 % | 11 s |
| U-Net | 470,977 | 4.1 % | 5.0 % | 10 min |
| U-Net without skips | 424,897 | 10.8 % | 12.5 % | 9 min |

- The MLP memorizes the training layouts (training error far below test
  error): it sees 1024 unrelated numbers and has to learn from 800 examples
  which elements are neighbours.
- The U-Net knows its neighbours through 3 × 3 convolutions and uses the same
  weights at every position, with a third of the MLP's weights.
- Without the skip connections both errors are high: pooling loses where
  exactly a fibre boundary is, and upsampling cannot bring it back.

`fem.py` takes about 7 s. Times are for an Apple M1 Max CPU (PyTorch 2.x);
`--device mps` (Apple GPU) brings the U-Net down to about 2 min, and `--device
cuda` works on an NVIDIA GPU or Google Colab. Other machines, devices and
PyTorch versions give slightly different numbers.

## Files

| file | content |
|---|---|
| `fem.py` | random fibre layouts, Q1 finite elements, von Mises stress at element centres |
| `models.py` | `MLP`, `UNet(skips=True/False)` |
| `train.py` | the training loop shared by all models (Adam, cosine learning rate, relative $L^2$ loss) |
| `figures.py` | data samples, predictions, training curves |

`fem.py` checks itself before generating data: a plate without fibres must
give $s = 1$ everywhere.
