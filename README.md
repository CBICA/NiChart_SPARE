# NiChart_SPARE

Implementation of SPARE scores calculation from Brain ROI Volumes ([NiChart_DLMUSE](https://github.com/CBICA/NiChart_DLMUSE)) and White-Matter-Lesion volumes ([NiChart_DLWMLS](https://github.com/CBICA/NiChart_DLWMLS)) as main features.

### Supported SPARE scores (as of June 2025):
- SPARE-CL : Any classfication
- SPARE-RG : Any regression
<!-- - SPARE-BA: Brain Age
- SPARE-AD: Alzheimer's 
- SPARE-HT: Hypertension
- SPARE-HL: Hyperlipidemia
- SPARE-T2B: Diabetes (Type 2)
- SPARE-SM: Smoking
- SPARE-OB: Obesity -->

## Installation

##### From GitHub
```bash
git clone https://github.com/CBICA/NiChart_SPARE.git
cd NiChart_SPARE
pip install -e .
```

## Argument Description

All arguments are passed as `-flag value`. Booleans are passed as the strings `True`/`False` (not Python booleans).

##### Core (needed for every run)

| Flag | Default | Meaning |
|---|---|---|
| `-a`, `--action` | *required* | What to do: `trainer` (fit a new model), `inference` (score data with an existing model), or `analysis` (reserved for future disease-effect analysis; not implemented yet). |
| `-t`, `--type` | *required* | Which SPARE score to compute. Currently functional end-to-end: `CL` (classification, e.g. disease vs. control) and `RG` (regression, e.g. brain age). `AD`/`BA` are accepted by early validation but are not yet wired to a training pipeline and will fail — use `CL`/`RG` for now. |
| `-i`, `--input` | *required* | Path to the input CSV. Must contain one row per subject, with ROI volume columns plus (for training) the target column. |
| `-mt`, `--model_type` | `SVM` | Which ML model family to use. `SVM` is the only one implemented; `MLP` is a recognized but not-yet-functional placeholder. |
| `-v`, `--verbose` | `0` | How much to print while running. `0` = quiet, higher numbers (1-3) print progressively more detail. |

##### SVM options

| Flag | Default | Meaning |
|---|---|---|
| `-sk`, `--svm_kernel` | `linear` | Which SVM kernel to fit: `linear`, `linear_fast` (quicker but more prone to bias), `poly`, `rbf`, or `sigmoid`. |
| `-bc`, `--bias_correction` | `1` | Only applies to regression (`-t RG`). Corrects the systematic tendency of regression models to underestimate high values and overestimate low ones. `0` disables it, `1` uses the Beheshti et al. method, `2` uses the Cole et al. method. |
| `-cb`, `--class_balancing` | `True` | Only applies to classification (`-t CL`). If `True`, re-weights classes during training so an imbalanced dataset (e.g. many more controls than patients) doesn't bias the model toward the majority class. |

##### ICV (intracranial volume) correction

| Flag | Default | Meaning |
|---|---|---|
| `-icv`, `--icv_correction` | `False` | If `True`, normalizes every ROI volume column by dividing it by the ICV column before training/inference — controls for differences in head/brain size across subjects. |
| `-icvc`, `--icv_column` | `DL_MUSE_Volume_702` | Name of the column in the input CSV that holds the ICV value, used only when `-icv True`. |

##### Training-only options

| Flag | Default | Meaning |
|---|---|---|
| `-ht`, `--hyperparameter_tuning` | `True` | If `True`, runs a grid search over model hyperparameters (e.g. SVM `C`) before fitting the final model. More accurate but slower. |
| `-tw`, `--train_whole` | `True` | If `True`, after tuning/cross-validation, refits one final model on the entire input dataset (recommended for the model you'll actually deploy). If `False`, only the CV fold models are kept. |
| `-cf`, `--cv_fold` | `5` | Number of cross-validation folds to use for evaluating model performance. Set to `0` to skip cross-validation entirely. |
| `-mo`, `--model_output` | *required for training* | File path where the trained model (a `.joblib` file containing the model plus preprocessing/metadata) will be saved. |
| `-tc`, `--target_column` | `target` | Name of the column in the input CSV holding the value to predict (e.g. diagnosis label for classification, age for regression). |

##### Inference-only options

| Flag | Default | Meaning |
|---|---|---|
| `-m`, `--model` | *required for inference* | Path to a previously trained `.joblib` model file to load. |
| `-o`, `--output` | *required for inference* | File path where the CSV of predicted SPARE scores will be written. |
| `--append-spare-tag` | *(none)* | Optional label to append to the output score column name, e.g. `--append-spare-tag ADNI` turns `SPARE_score` into `SPARE_ADNI`. Useful when chaining multiple SPARE runs into one pipeline. |

##### Shared data-handling options

| Flag | Default | Meaning |
|---|---|---|
| `-kv`, `--key_variable` | `MRID` | Name of the column that uniquely identifies each subject/row (e.g. `MRID`, `ID`, `PTID`). Used to match rows across files and is dropped from the model features. |
| `-ic`, `--ignore_column` | *(none)* | Comma-separated list of extra column names to exclude from the model features (e.g. `Study,SITE,Sex`), on top of the key variable and target column. |
| `-di`, `--disease` | `AD` | Name of a column indicating disease status. Currently accepted but not consumed by `trainer`/`inference` — reserved for the upcoming `analysis` action. |

Run `NiChart_SPARE -h` at any time to see this list generated directly from the installed version.


### Example Usage
##### Training a classifier
```bash
NiChart_SPARE -a trainer \
              -t CL \
              -i training_input.csv \
              -mt SVM \
              -sk linear \
              -ht True \
              -tw True \
              -cf 5 \
              -mo output_model.joblib \
              -kv MRID \
              -tc clf_target_column_name \
              -ic Study,SITE,Sex \
              -cb False \
              -v 1
```
##### Training a regressor
```bash
NiChart_SPARE -a trainer \
              -t RG \
              -i training_input.csv \
              -mt SVM \
              -sk linear \
              -ht False \
              -tw True \
              -bc True \
              -cf 5 \
              -mo output_model.joblib \
              -kv MRID \
              -tc rg_target_column_name \
              -ic Study,SITE,Sex \
              -v 1
```
##### Inference
```bash
NiChart_SPARE -a inference \
              -t RG \
              -i test_input \
              -m model.joblib \
              -o test_output.csv \
              -kv MRID
                
```

```bash
NiChart_SPARE -a inference \
              -t CL \
              -i test_input \
              -m model.joblib \
              -o test_output.csv \
              -kv MRID
                
```
## Model File (`.joblib`) Schema

A model saved by `trainer` (`-mo model.joblib`) is a single `joblib`-pickled Python `dict` (see `save_svm_model`/`load_svm_model` in `NiChart_SPARE/svm.py`) with five top-level keys:

```
{
  "model": {
    "model": <fitted sklearn estimator>,   # LinearSVR/SVR (RG) or LinearSVC/SVC (CL)
    "bias": <dict or None>                 # regression-only bias correction, see below
  },
  "meta_data": { ... },        # see below
  "preprocessor": { ... },     # see below
  "hyperparameter_tuning": { ... },  # see below
  "cross_validation": { ... }  # see below
}
```

`__main__.py` reads `model['meta_data']['model_description']['model_type']` directly off the loaded dict to route inference, and `svm.load_svm_model` unpacks all five keys as a tuple.

##### `meta_data` (built by `get_metadata` in `util.py`)

```
{
  "spare_type": "CL" | "RG",
  "package_version": "0.1.0",
  "model_description": {
    "model_type": "SVM",
    "kernel": "linear" | "linear_fast" | "poly" | "rbf" | "sigmoid"
  },
  "training_data_description": {
    "target_column": str,
    "feature_names": [str, ...],   # all input columns except the target
    "feature_count": int,
    "data_size": int               # number of training rows
  },
  "pipeline_description": {
    "hyperparameter_tuning": bool,
    "cv_fold": int,
    "model_class_balancing": bool,
    "trained_using_whole_set": bool
  }
}
```

`infer_svm_model` uses `training_data_description.feature_names`/`target_column` to validate and subset an inference CSV before predicting.

##### `preprocessor` (built by `get_preprocessors` in `util.py`)

```
{
  "feature_encoder": dict[str, LabelEncoder] | None,  # one entry per categorical column
  "feature_scaler": dict[str, StandardScaler] | None   # one entry per numeric column
}
```

##### `model.bias` (regression only; set in `pipelines/spare_svm_regression.py`, `None` for classification)

```
{
  "model": <fitted sklearn LinearRegression>,
  "method": 1 | 2,       # 1 = Beheshti et al. (residual approach), 2 = Cole et al.
  "intercept": float,
  "coef": float
}
```

##### `hyperparameter_tuning` (built by `get_hyperparameter_tuning`; `{}` if `-ht False`)

```
{
  "hyperparameter_tuner": <fitted sklearn GridSearchCV>,
  "best_params": dict,      # winning parameter combination
  "search_grid": dict       # full grid that was searched
}
```

##### `cross_validation` (`{}` if `-cf 0`; shape differs by `spare_type`)

Regression (`RG`) — one repeat, keyed by fold:
```
{
  "Repeat_0": {
    "scores": {"Fold_0": {<metric>: float, ...}, "Fold_1": {...}, ...},
    "cv_results": {"Fold_0": <pandas.DataFrame: test_reference, test_prediction, test_prediction_BC?, fold>, ...}
  }
}
```

Classification (`CL`) — 3 repeats, keyed by fold, with a per-fold fitted model:
```
{
  "Repeat_0": {
    "Fold_0": {
      "model": <fitted sklearn estimator for this fold>,
      "cv_validation": <pandas.DataFrame: test_reference, test_prediction, test_decision_function, fold>,
      "cv_scores": {<metric>: float, ...}
    },
    "Fold_1": { ... }, ...
  },
  "Repeat_1": { ... },
  "Repeat_2": { ... }
}
```

`infer_svm_model` uses these per-fold models (when `cv_fold > 1`) to additionally emit a `SPARE_<type>_Fold_<n>` prediction column alongside the main model's output.

## Documentation

Coming Soon (Wiki-page)

Supported SVM kernels ("-k" or "--kernel" argument):
- linear_fast (prone to bias)
- linear
- rbf
- poly

## Publications

- SPARE-BA

  Habes, M. et al. Advanced brain aging: relationship with epidemiologic and genetic risk factors, and overlap with Alzheimer disease atrophy patterns. Transl Psychiatry 6, e775, [doi:10.1038/tp.2016.39](https://doi.org/10.1038/tp.2016.39) (2016).

- SPARE-AD

  Davatzikos, C., Xu, F., An, Y., Fan, Y. & Resnick, S. M. Longitudinal progression of Alzheimer's-like patterns of atrophy in normal older adults: the SPARE-AD index. Brain 132, 2026-2035, [doi:10.1093/brain/awp091](https://doi.org/10.1093/brain/awp091) (2009).

- diSPARE-AD

  Hwang, G. et al. Disentangling Alzheimer's disease neurodegeneration from typical brain ageing using machine learning. Brain Commun 4, fcac117, [doi:10.1093/braincomms/fcac117](https://doi.org/10.1093/braincomms/fcac117) (2022).

- SPARE-CVMs (HT, HL, T2B, SM, OB)

  Govindarajan, S.T., Mamourian, E., Erus, G. et al. Machine learning reveals distinct neuroanatomical signatures of cardiovascular and metabolic diseases in cognitively unimpaired individuals. Nat Commun 16, 2724, [doi:10.1038/s41467-025-57867-7](https://doi.org/10.1038/s41467-025-57867-7) (2025). 


  # Notes
  - data_prep.py : subsets data for training/cross-validation and also perform additional processes including standardscaling, adjustment of Age/Sex/ICV effects 

  - pipelines/spare_(BIOMARKER).py : (Biomarker) specific training pipelines

  - __main__.py : entry point for CLI, handle input arguments and calling of specific spare training pipeline or inferencing code