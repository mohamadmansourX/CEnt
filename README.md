<div align="center">

# CEnt: An Entropy-based Model-agnostic Explainability Framework to Contrast Classifiers’ Decisions

Official Implementation of CEnt

</div>

<p align="center">
  <img src="docs/method.png" height="160">
</p>

## Abstract

In this work, the authors present a new framework termed BEVFormer, which learns unified BEV representations with spatiotemporal transformers to support multiple autonomous driving perception tasks. In a nutshell, BEVFormer exploits both spatial and temporal information by interacting with spatial and temporal space through predefined grid-shaped BEV queries. To aggregate spatial information, the authors design a spatial cross-attention that each BEV query extracts the spatial features from the regions of interest across camera views. For temporal information, the authors propose a temporal self-attention to recurrently fuse the history BEV information.
The proposed approach achieves the new state-of-the-art **56.9\%** in terms of NDS metric on the nuScenes test set, which is **9.0** points higher than previous best arts and on par with the performance of LiDAR-based baselines.

## Getting the code

Getting the Source Code

```shell
git clone ...
cd CENT/
```

## Installing Dependencies

```shell
bash install.sh
```

Or, if you prefer to using conda:

```shell
conda env create -f environment.yml
```

## Experiments Results

### Tabular Data

CEnt improvement in proximity, latency, and attainability with no constraint violation compared to previous methods

<p align="center">
  <img src="docs/tabular_data.png" >
</p>

### Images Data

CEnt also was also able to achieve great and very potential results in the image applications as seen in the case of mnits.

CEnt was able to better understand the image and trace the pixels to flip the class instead of going to a whole new drawing as in CEM.

<p align="center">
  <img src="docs/cent_vs_cem.png" >
</p>

### Text Data

CEnt was also tested on text data and was able to flip the predictions.  CEnt can then serve as a debugging tool that highlights vulnerabilities in the context of adversarial attacks.

<p align="center">
  <img src="docs/nlp.png" >
</p>

## Running the Benchmarks


```shell
python becnhmark_experiment.py
```

For each dataset the following will be created:

```shell
$tree outputs/

outputs/
└── adult
    ├── bench_csvs
    │   ├── cent_tensorflow_ann_bench.csv
    │   ├── cent_tensorflow_ann_counterfactuals.csv
    │   ├── cent_tensorflow_ann_DTScores.csv
    │   ├── cent_tensorflow_ann_factuals.csv
    │   ├── dice_tensorflow_ann_bench.csv
    │   ├── dice_tensorflow_ann_counterfactuals.csv
    │   ├── dice_tensorflow_ann_factuals.csv
    │   ├── growing_spheres_tensorflow_linear_bench.csv
    │   ├── growing_spheres_tensorflow_linear_counterfactuals.csv
    │   └── growing_spheres_tensorflow_linear_factuals.csv
    │   └──   .
    │   └──   .
    │   └──   .
    ├── benchmark_results.csv
    ├── checks.csv
    ├── loss_plot.png
    ├── loss_plot_steps.png
    ├── models_logs.txt
    └── model_zoo_metrics.csv
└── compas
  .
  .
  .

```

- `model_zoo_metrics.csv`: contains the metrics of the black-box models used in the experiments (common between resource for each dataset).
- `models_logs.txt`: contains the logs of the black-box models used trainings.
- `benchmark_results.csv`: contains the dataframe of the results of the experiments per dataset using the 9 implemented metrics.
- `checks.csv`: contains info of what resource failed or ran successfully.
- `loss_plot.png`: plot of the loss of vae benchmark per step.
- `loss_plot_steps.png`: plot of the loss per epoch
- `bench_csvs`: contains the csv files results per recourse per model type e.g. for dataset `adult`, resource `cent`, model type `ann` there will be {factuals: `cent_tensorflow_ann_bench.csv`}, {counterfactuals: `cent_tensorflow_ann_counterfactuals.csv`}, and {benchmarks: `cent_tensorflow_ann_bench.csv`} csv files.

