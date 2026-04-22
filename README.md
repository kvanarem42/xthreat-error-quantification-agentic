# Model Quality in Football: Quantifying the Quality of an Expected Threat Model

<div align="center">

[![ArXiv](https://img.shields.io/badge/arXiv-preprint-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org)
[![Python](https://img.shields.io/badge/Python-3.12.3-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![StatsBomb](https://img.shields.io/badge/Data-StatsBomb%20Open%20Data-0066CC?style=flat-square)](https://github.com/statsbomb/open-data)

*Koen van Arem · Jakob Söhl · Mirjam Bruinsma · Geurt Jongbloed*

*Delft University of Technology*

</div>

---

## Overview

This repository contains the data, code, and analysis accompanying the preprint:

> **"Model quality in football: quantifying the quality of an Expected Threat model"**
> Koen van Arem, Jakob Söhl, Mirjam Bruinsma, Geurt Jongbloed
> *(ArXiv link to be added)*

Expected Threat (xT) models are widely used in football analytics to value ball actions. This work addresses a fundamental but underexplored question: **how good is a trained xT model, really?** We develop a framework to rigorously quantify model quality, provide uncertainty estimates, and derive practical rules of thumb for practitioners.


---

## Repository Structure

```
.
├── xThreat.py                                     ← Core xT model class
├── requirements.txt
│
├── 1-data-preparation/
│   ├── download_clean_join.ipynb                  ← Main data pipeline notebook
│   ├── datacleaner.py
│   ├── datadownloader.py
│   └── datasetcreator.py
│
├── 2-train-models/
|   ├── sample_distribution.py                                  ← Example script to resample model
│   ├── true-models/
│   │   ├── calculate_true_models.py                            ← Script to precalculate the 'true' models.
│   ├── resampled-models/model-error-distribution/
│   │   ├── sample_distribution_parallel.py                     ← Scripts for running simulations in section s4.1
│   │   └── sample_distribution_parallel_extended.py            ← Scripts for running simulations in section s4.1
|   ├── resampled-models/model-error-distribution/              ← Missing directory with scripts for running simulations in section s4.1
│   └── model-storage/resampled-models/
│       ├── xt-N100000-n_x16-n_y12-i_bootstrap42-...pickle      ← Example resampled models
│       └── xt-N1300000-n_x64-n_y48-i_bootstrap169-...pickle    ← Example resampled models
│
├── 3-calculate-values/
│   ├── s4.1-model-error-distribution/
│   │   ├── calculate_values_normal_xthreat.py                  ← Scripts to calculate quantities from resampled models section s4.1.1
│   │   ├── calculate_values_normal_xthreat_extended.py         ← Scripts to calculate quantities from resampled models section s4.1.1
│   │   ├── bootstrap_errors_normal_xthreat.csv                 ← Results containing these quantities (Part 1/2)
│   │   └── bootstrap_errors_normal_xthreat_extended.csv        ← Results containing these quantities (Part 2/2)
│   └── s4.2-max-error-quartile-changes/
│       ├── calc-model-error/
│       │   ├── calculate_values_normal_xthreat_max_error_full.py     ← Script to calculate quantities from resampled models section s4.1.2
│       │   └── bootstrap_errors_normal_xthreat_max_error_full.csv    ← Results containing these quantities
│       └── calc-player-xT-created/
│           ├── calculate_resampled_player_ratings.py                 ← Script calculating player ratings from resampled models
│           └── calculate_values_normal_xthreat.py                    ← Script calculating player ratings from ground truth models
│
└── 4-investigate-results/
    ├── s4.1-distribution-model-error/
    │   ├── exploratory_analysis.ipynb              ← Notebook showing some extra visualizations
    │   ├── distribution_fitting.ipynb              ← Notebook fitting the distribution (S4.2.1)
    │   └── model_parameters_influence.ipynb        ← Notebook showing influence of model parameters (S4.2.1)
    ├── s4.2-acceptable-model-error/
    │   └── find_maximal_error.ipynb                ← Notebook calculating maximal acceptable error (S4.2.2)
    └── s5-application-and-illustration/
        ├── example_euros_2020.ipynb                ← Notebook illustrating application of xT values (S5.2)
        ├── rules_of_thumb_paper.ipynb              ← Notebook illustrating rules of thumb (S5.1)
        └── xthreat_explanation.ipynb               ← General visualisations for elements in the xT model (S2)
```

---

## `xThreat.py` — Core Model Class

The backbone of this repository. `xThreat.py` implements the `XThreat` model class used across nearly all scripts and notebooks. It provides the main interface to **train, visualize, and apply** an Expected Threat model, and is a natural starting point if you want to understand or reuse the modelling framework.

---

## Part 1 — Data Preparation

📁 `1-data-preparation/`

| File | Description |
|------|-------------|
| `download_clean_join.ipynb` | End-to-end pipeline: download, clean, and join StatsBomb event data |
| `datadownloader.py` | Fetches raw event data from the StatsBomb open dataset |
| `datacleaner.py` | Cleans and filters raw events |
| `datasetcreator.py` | Joins cleaned data into a unified modelling-ready format |

Run `download_clean_join.ipynb` first to produce the dataset used in all subsequent steps.

---

## Part 2 — Model Training

📁 `2-train-models/`

| Subfolder | Description |
|-----------|-------------|
| `true-models/` | Scripts to train the ground-truth xT models on the full dataset |
| `resampled-models/model-error-distribution/` | Scripts to sample new models from the ground truth (parallel versions for HPC) |
| `model-storage/resampled-models/` | Two example resampled models included for illustration (`.pickle`) |

> **⚠️ Note on large-scale computations**
>
> The full resampling procedure was run on the **[DelftBlue supercomputer](https://www.tudelft.nl/dhpc/ark:/44463/DelftBluePhase1)** at TU Delft and produces a large number of model files not included here due to storage constraints. Two example models are provided. The sampling scripts can be adapted to run locally at smaller scale.
>
> If you are interested in the full set of resampled models, feel free to reach out to [k.w.vanarem@tudelft.nl](mailto:k.w.vanarem@tudelft.nl) to discuss transfer options.

---

## Part 3 — Calculate Values

📁 `3-calculate-values/`

Contains the core computations that analyse the resampled models. Subfolders are labelled to match the paper's section numbering.

| Subfolder | Paper section | Description |
|-----------|---------------|-------------|
| `s4.1-model-error-distribution/` | §4.1.1 | Computes bootstrap error distributions; pre-computed results saved as `.csv` |
| `s4.2-max-error-quartile-changes/calc-model-error/` | §4.1.2 | Computes maximum model errors across configurations; results in `.csv` |
| `s4.2-max-error-quartile-changes/calc-player-xT-created/` | §4.1.2 | Computes resampled player ratings |

> **⚠️ Note on player ratings**
>
> The computed player rating values from `calc-player-xT-created/` are **not included** in the repository due to file size. Please contact [k.w.vanarem@tudelft.nl](mailto:k.w.vanarem@tudelft.nl) if you need these files.

---

## Part 4 — Investigate Results

📁 `4-investigate-results/`

> 🌟 **Recommended starting point for most visitors**

Notebooks that present the paper's findings. Subfolders correspond to paper sections.

### §4.1 — Distribution of Model Error
📁 `s4.1-distribution-model-error/`

| Notebook | Description |
|----------|-------------|
| `distribution_fitting.ipynb` | Fits parametric distributions to the bootstrap errors |
| `model_parameters_influence.ipynb` | Analyses how grid resolution and sample size affect model quality |
| `exploratory_analysis.ipynb` | First look at the error distributions across model configurations |

### §4.2 — Acceptable Model Error
📁 `s4.2-acceptable-model-error/`

| Notebook | Description |
|----------|-------------|
| `find_maximal_error.ipynb` | Determines the threshold for acceptable model error in terms of quartile changes in player rankings |

### §5 — Application and Illustration
📁 `s5-application-and-illustration/`

| Notebook | Description |
|----------|-------------|
| `rules_of_thumb_paper.ipynb` | Derives and demonstrates practical rules of thumb for model quality |
| `example_euros_2020.ipynb` | Applied example: using the xT model on UEFA Euro 2020 data |
| `xthreat_explanation.ipynb` | Visualizations of part of the xT model |

---

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Recommended Workflow

```
1-data-preparation  ──►  2-train-models  ──►  3-calculate-values  ──►  4-investigate-results
```

Most visitors can jump straight to **Part 4** — the notebooks allow for a quick visual inspection of the methods

---

## Authors & Contact

| Author | Affiliation | Contact |
|--------|-------------|---------|
| **Koen van Arem** | Delft University of Technology | [k.w.vanarem@tudelft.nl](mailto:k.w.vanarem@tudelft.nl) |
| **Jakob Söhl** | Delft University of Technology | — |
| **Mirjam Bruinsma** | AFC Ajax | — |
| **Geurt Jongbloed** | Delft University of Technology | — |

For questions about large data files (resampled models, player ratings), please contact Koen van Arem at [k.w.vanarem@tudelft.nl](mailto:k.w.vanarem@tudelft.nl).

---

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{vanArem2024xthreat,
  title   = {Model quality in football: quantifying the quality of an Expected Threat model},
  author  = {van Arem, Koen and S{\"o}hl, Jakob and Bruinsma, Mirjam and Jongbloed, Geurt},
  journal = {arXiv preprint},
  year    = {2026},
  note    = {arXiv link to be added}
}
```

---

## Data

This project uses the **[StatsBomb Open Data](https://github.com/statsbomb/open-data)** dataset. Please comply with StatsBomb's [terms of use](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf) when using this repository.

---

<div align="center">
<sub>Made with ⚽ at Delft University of Technology</sub>
</div>