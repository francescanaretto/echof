# EchoForest: how to

In the following a quick description of the steps for use the EchoForest. For reproducing the results of the paper, see REPRODUCIBILITY.md (contains execution details, model selection settings etc.).

## Required Python libraries

The main workflows rely on the following libraries.

- `numpy`
- `pandas`
- `scikit-learn`
- `torch`
- `matplotlib`
- `seaborn`
- `sdv`

For running the codes in `competitors/`, additional libraries and dependencies are needed.

Minimal working setup:

```bash
python -m pip install numpy pandas scikit-learn torch matplotlib seaborn sdv
```

## Main pipeline of EchoForest:


1. Train an original black-box model on real data (NN in our example)
2. Generate synthetic data by using our method that queries the black-box
3. Train the synthetic Random Forest surrogate, referred to as **EchoForest**
4. Extract modular explanations from EchoForest (a demo is available at: https://anonymous.4open.science/w/demo-echof-4C80/)

All scripts keep a default configuration in the file, but the synthetic-data
generators also support command-line overrides for the main run parameters.

## 1. Train the original black-box on real data

Neural network black-box:

```bash
python Code-EchoForest/blackboxes/NN-original.py
```

The script trains the original neural-network black-box and performs explicit
hyperparameter search:

- `NN-original.py` uses repeated stratified cross-validation over a manual grid
  of MLP configurations

## 2. Generate synthetic data

The main synthetic data generation is:

```bash
python Code-EchoForest/generation/generate_echo_forest_data.py --datasets income spotify --n-synth 80000 --mode entropy25 --guiding-bb nn
```

This step queries the original black-box and generates synthetic labeled data.

Main options (cli):

- `--datasets income spotify`
- `--n-synth 80000`
- `--mode entropy25 | entropy50 | margin | kappa | logit`
- `--guiding-bb nn | rf`

If differential privacy is needed at query time, the DP variant is:

```bash
python Code-EchoForest/generation/generate_echo_forest_data_dp.py --datasets income --n-synth 80000 --mode logit --guiding-bb nn --epsilon 1.0 --noise laplace
```

Additional DP options (cli):

- `--epsilon 0.5`
- `--noise laplace | gaussian`

## 3. Train EchoForest

After synthetic data has been produced, train the Random Forest surrogate
(EchoForest) with:

```bash
python Code-EchoForest/generation/train_echo_forest.py
```

This script trains the Random Forest surrogate on synthetic data.

For DP-query runs, the corresponding training script is:

```bash
python Code-EchoForest/generation/train_echo_forest_dp.py
```

## 4. Validate EchoForest

```bash
python Code-EchoForest/validation/validation.py
```

Its configuration section can be adjusted to evaluate the desired dataset and
surrogate setting, such as the metric, the fidelity etc.

Additional validation summaries:

```bash
python Code-EchoForest/validation/nn_prediction_performance.py
python Code-EchoForest/validation/echo_forest_structure_summary.py
```

## 5. Extract explanations from EchoForest

To extract and select supporting rules from EchoForest:

```bash
python Code-EchoForest/explainability/select_supporting_rules.py
```

This step produces rule-level outputs, including:

- full supporting-rule sets
- compact top-k selections (k is user defined)
- per-instance summaries

To export compact explanation views:

```bash
python Code-EchoForest/explainability/export_modular_explanations.py
```

To evaluate the extracted explanations:

```bash
python Code-EchoForest/explainability/evaluate_rule_explanations_binary.py
```
multiclass settings:

```bash
python Code-EchoForest/explainability/evaluate_rule_explanations_multi.py
```

