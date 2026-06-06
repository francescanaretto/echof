# Reproducibility

## Environment information

- Python environment:
  `Python 3.10` in a Conda environment
- Main libraries:
  `numpy`, `pandas`, `scikit-learn`, `torch`, `matplotlib`, `seaborn`, `sdv`
- Operating system:
  `Linux 5.15.0-112-generic`, `x86_64`
- CPU:
  `AMD EPYC 9554 64-Core Processor`
- Sockets:
  `2`
- Cores per socket:
  `64`
- Threads per core:
  `2`
- Total logical CPUs:
  `256`
- RAM:
  `1.5 TiB`

## Parallelism

- neural-network black-box training:
  `torch.set_num_threads(10)` and `torch.set_num_interop_threads(10)`
- EchoForest training:
  `n_jobs = 20`
- DP EchoForest training:
  `n_jobs = 20`

## Random seeds

- neural-network black-box:
  `42`
- EchoForest training:
  `42`

## Data sources

For space reasons, we do not report all datasets used in the experiments here.
Datasets come from publicly available sources such as Hugging Face,
Folktables, or other public repositories.

The following sources are especially relevant for reproducing the datasets
that required custom handling in this release:

- `income`:
  [socialfoundations/folktables](https://github.com/socialfoundations/folktables)
  using only states `WA`, `NY`, `OH`, `CA`, `MN`, and `NC`, and years `2014`
  and `2015`
- `landsat` and related variants:
  [mstz/landsat](https://huggingface.co/datasets/mstz/landsat)
- `segment`:
  [mstz/segment](https://huggingface.co/datasets/mstz/segment)
- `pol`:
  [mstz/pol](https://huggingface.co/datasets/mstz/pol)
- `activity`:
  [PAMAP2 Physical Activity Monitoring](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)

## Model selection

### Original neural-network black-box

Script:

- `Code-EchoForest/blackboxes/NN-original.py`

Selection procedure:

- repeated stratified cross-validation over a manual hyperparameter grid
- selection metric:
  balanced accuracy

Cross-validation:

- `RepeatedStratifiedKFold`
- `n_splits = 5`
- `n_repeats = 2`

Hyperparameter grid:

- hidden layers:
  `(10, 5)`, `(5,)`, `(16, 8)`
- dropout:
  `0.1`, `0.2`, `0.3`
- learning rate:
  `1e-3`, `5e-4`
- weight decay:
  `1e-4`, `5e-4`, `1e-3`
- batch size:
  `32`, `64`
- epochs:
  `50`
- early-stopping patience:
  `3`

### EchoForest training

Script:

- `Code-EchoForest/generation/train_echo_forest.py`

Selection procedure:

- exhaustive `ParameterGrid`
- the model is trained on synthetic data
- model selection is performed against original data relabeled by the black-box
- selection metric:
  macro F1

Main grid:

- `n_estimators`: `50`, `70`, `100`, `150`, `200`
- `max_depth`: `None`, `5`, `10`, `15`, `20`

### DP EchoForest training

Script:

- `Code-EchoForest/generation/train_echo_forest_dp.py`

Selection procedure:

- exhaustive `ParameterGrid`
- the model is trained on DP-query synthetic data
- model selection is performed against original data relabeled by the black-box
- selection metric:
  macro F1

Main grid:

- `n_estimators`: `50`, `70`, `100`, `150`, `200`
- `max_depth`: `None`, `5`, `10`, `15`, `20`

## Measured runtimes

### Synthetic data generation

The code records timing information for synthetic data generation in the timing
CSV files written by:

- `Code-EchoForest/generation/generate_echo_forest_data.py`
- `Code-EchoForest/generation/generate_echo_forest_data_dp.py`

Recorded timing fields:

- `time_generate_s`
- `time_label_s`
- `time_save_s`
- `time_total_s`

### Rule explanation extraction

The code records timing information for rule extraction and rule selection
through:

- `Code-EchoForest/explainability/analyze_prems_on_posthoc_instances.py`

Recorded timing fields:

- `time_extract_seconds`
- `time_select_seconds`
- `time_total_seconds`

## Runtime of Echo Forest

| Stage of EchoForest             |   Runtime (in sec) |
|---------------------------------|--------------------:|
| Synthetic generation (standard) | `1007.65 ± 2147.17` |
| Synthetic generation (DP-query) |  `1014.97 ± 671.62` |
| Rule extraction (standard)      |     `0.023 ± 0.011` |
| Rule selection                  |     `0.001 ± 0.000` |
| Extraction + selection          |     `0.023 ± 0.011` |



We omit the training times of neural networks and Random Forests, as they depend heavily on implementation details, the size of the hyper-parameter search space, the number of available CPU threads, and, when applicable, GPU availability.
