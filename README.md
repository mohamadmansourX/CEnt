<div align="center">

# CEnt: An Entropy-based Model-agnostic Explainability Framework to Contrast Classifiers’ Decisions

Official Implementation of CEnt.
>based on the [CARLA Framework](https://github.com/carla-recourse/CARLA)

</div>

<p align="center">
  <img src="docs/method.png" height="160">
</p>

## Abstract

In this work, we presents CEnt, a novel entropy-based method, that supports an individual facing an undesirable outcome under a decision- making system with a set of actionable alternatives to im- prove their outcome. CEnt samples from the latent space learned by VAEs and builds a decision tree augmented with feasibility constraints. Graph search techniques are then em- ployed to find a compact set of feasible feature tweaks that can alter the model’s decision.

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

