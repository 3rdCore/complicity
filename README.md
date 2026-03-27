# A Compression Perspective on Simplicity Bias

Code for the paper *A Compression Perspective on Simplicity Bias*

We study whether **two-part optimal compression** (via the Minimum Description Length principle) serves as a predictive theory of neural-network behaviour under different data-regime. The key insight is that simplicity bias is not a fixed property of a learner but a *data-dependent* preference: as the amount of training data grows, the MDL-optimal compressor shifts from simple spurious shortcuts to more predictive (but more complex) coding scheme.  We validate this prediction on a controlled semi-synthetic benchmark derived from Colored MNIST.

---

## Setup

```bash
git clone https://github.com/3rdCore/complicity.git
cd complicity
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install .
python scripts/download.py --download cmnist
```

The last command downloads EMNIST (used as the digit backbone) into `data/benchmark/emnist/`.

---

## Reproducing the experiments

### Interactively (Jupyter / VS Code)

Open `main.ipynb` and run all cells. Make sure to set `DEBUG_MODE = False`.  The notebook reads experiment parameters from environment variables; defaults match the paper's **Scenario A** (spurious colour vs. robust digit).

To switch to **Scenario B** (robust digit vs. complex watermark):

```bash
EXPERIMENT_SETTING=2 WATERMARK_BANK_SIZE=50 jupyter notebook main.ipynb
```

### On a SLURM cluster — single run

Edit the hyperparameters at the top of `scripts/run_single_notebook.sh`, then submit:

```bash
sbatch scripts/run_single_notebook.sh
```

Results (executed notebook + CSV files) are written to a timestamped subfolder of `results/`.

### On a SLURM cluster — full sweep (paper figures)

`run_multiple_notebook.sh` submits one SLURM job per parameter combination:

```bash
bash scripts/run_multiple_notebook.sh
```

The sweep covers the three key task characteristics:

| Figure | Variable swept | Script variable |
|--------|----------------|-----------------|
| 3a — robust feature predictiveness | label-flip probability | `FLIP_PROB` |
| 3a — spurious feature predictiveness | spurious correlation strength | `SPUR_PROB` |
| 3b — bayes feature complexity | watermark bank size | `WATERMARK_BANK_SIZE` |

Edit the `CONFIGS_TO_RUN` variable and the `case` blocks in `run_multiple_notebook.sh` to define which parameter grids to sweep.

### Generating the paper figures

Once all jobs have completed, open and run `plot.ipynb`. Set `experiment_path` to the results folder. The notebook aggregates the CSV outputs and reproduces the paper figures as PDFs in `results/plot/`.

---

## Repository structure

```
main.ipynb                     # Main experiment notebook
plot.ipynb                     # Plotting notebook (paper figures)
source/
├── datasets.py                # CMNIST dataset with colour / watermark generation
├── algorithms.py              # ERM training loop
├── networks.py                # Featurizers and linear classifiers
└── utils/
    ├── eval_helper.py         # Metrics, PCL envelope utilities
    ├── hparams_registry.py    # Default hyperparameters
    ├── misc.py                # General utilities
    ├── notebook_helpers.py    # PCL curve computation, permutation tests
    └── plotting.py            # Standardized plot functions
scripts/
├── run_single_notebook.sh     # SLURM job script (single run)
├── run_multiple_notebook.sh   # SLURM sweep script (submits one job per config)
└── download.py                # Data download helper
```

---
## Citation

```bibtex
@article{marty2024compression,
  title={A Compression Perspective on Simplicity Bias},
  author={Marty, Tom and Elmoznino, Eric and Gagnon, Leo and Kasetty, Tejas and Nishikawa-Toomey, Mizu and Mittal, Sarthak and Lajoie, Guillaume and Sridhar, Dhanya},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```
