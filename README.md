# A Unified Pairwise Distribution Matching Framework for Graph Domain Adaptation under Structure Shift

Code to reproduce the numerical results in the paper.

## Repository layout

```text
run.py          # Main CLI: csbm / csbm-calib / real
methods/        # Downstream GDA backbones (a2gnn, specreg, dgsda, sagda)
models/         # Reweighting / predictor modules (plsa, pairalign, strurw, ...)
utils/          # Data loading, calibration, metrics, graph utilities
data/           # Datasets (not included; see "Datasets" below)
```

## Installation

Developed and tested with Python 3.9.

```bash
conda create -n gda python=3.9 -y
conda activate gda
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.6.1
pip install -r requirements.txt
```

## Datasets

```text
data/acm/raw/        acm_docs.txt   acm_edgelist.txt   acm_labels.txt
data/dblp/raw/       dblp_docs.txt  dblp_edgelist.txt  dblp_labels.txt
data/cora_data/      good_cora_src_data.npz   good_cora_src_adj.npz
                     good_cora_tar_data.npz   good_cora_tar_adj.npz
data/twitch/         twitch_DE.npz    twitch_ENGB.npz    twitch_ES.npz   ...
data/airport_data/   airport_usa.npz  airport_brazil.npz  airport_europe.npz
```

- **ACM / DBLP**: citation-network splits from Wu et al., *Unsupervised
  domain adaptive graph convolutional networks*, WWW 2020.
- **Twitch**: [Twitch Social Networks](https://snap.stanford.edu/data/twitch-social-networks.html).
- **CORA**: Bojchevski & Günnemann (2018), using the Degree-split from
  Gui et al., *GOOD: A graph OOD benchmark*, NeurIPS 2022.
- **Airport**: Zhu et al. (2021b) flight-connection networks,
  [github.com/GentleZhu/EGI](https://github.com/GentleZhu/EGI/tree/main/data).

Place the files as shown above.

## Usage

### Real-data experiments (Tables 1-4)

`real` runs one hyperparameter setting for a given adaptation direction, GDA
backbones (`--methods`), structure-correction models (`--models`), and seeds.
Results are saved as JSON lines in `real_results/` (or the path given by `--out`).

```bash
python run.py --device cuda real \
  --dataset acm_dblp --src acm --tar dblp \
  --methods a2gnn specreg dgsda sagda \
  --models plain plsa pairalign strurw \
  --seeds 0 1 2 3 4 \
  --mmd-lam 0.1 --lam-w 0.0 --lam-plsa 0.0 --smooth-T 1.0 --lam-pair 0.0 --lam-blend 1.0
```

The other settings in the paper are

```text
Table 1  --dataset acm_dblp --src acm --tar dblp     |  --dataset acm_dblp --src dblp --tar acm
Table 2  --dataset twitch   --src DE  --tar ENGB     |  --dataset twitch   --src ENGB --tar DE
Table 3  --dataset cora     --k-labels 6             |  --dataset cora     --k-labels 8
Table 4  --dataset airport  --src usa --tar brazil   |  --dataset airport  --src usa  --tar europe
```

The reported results use the hyperparameter grid below (Appendix B.3 of the
paper). For each adaptation direction, backbone, model, and seed, we run every
combination, select the one with the highest `val_acc`, and report its
`test_acc`. Here `val_acc` is the accuracy on a labeled validation split of
20% of the target nodes, and `test_acc` is the accuracy on the remaining 80%.

| Flag          | Values               | Models                   |
|---------------|----------------------|--------------------------|
| `--mmd-lam`   | 0.01, 0.1, 1.0       | all                      |
| `--lam-w`     | 0, 0.1, 1.0          | plsa, pairalign          |
| `--lam-plsa`  | 0, 0.01, 0.1, 0.5    | plsa                     |
| `--smooth-T`  | 0.1, 0.5, 1.0, 2.5   | plsa                     |
| `--lam-pair`  | 0, 0.1, 1, 10        | pairalign                |
| `--lam-blend` | 0.2, 0.6, 1.0        | plsa, pairalign, strurw  |

`real` also has two optional flags for label shift. Both are off by default.

- `--mmd-n-iters K`: alternates MLLS label-shift estimation and feature
  alignment for `K` rounds, sampling source nodes in the MMD term with
  probability proportional to the estimated weights.
- `--iw-erm`: weights the source classification loss of the GDA backbone by
  the final MLLS weights.

### Synthetic experiments

```bash
# Table 5 -- calibration ablation
python run.py --device cuda csbm-calib --seeds 0 1 2 3 4 --methods plsa_cal plsa_uncal pairalign

# Figure 1 -- CSBM importance-weight estimation error
python run.py --device cuda csbm --mode fig1 --L 2 --n 5000 --seeds 0 1 2 3 4 --methods pairalign plsa

# Figure 2 -- estimation error vs. graph size
python run.py --device cuda csbm --mode fig2 --L 2 --seeds 0 1 2 3 4 --methods pairalign plsa
python run.py --device cuda csbm --mode fig2 --L 3 --seeds 0 1 2 3 4 --methods pairalign plsa
```

## License

MIT License, see `LICENSE`.