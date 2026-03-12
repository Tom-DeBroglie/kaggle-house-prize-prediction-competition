# Kaggle House Prices Prediction Competition

## Overview

This repository contains my first Kaggle competition project: predicting house prices using machine learning. The project leverages Python-based data processing, feature engineering, and gradient boosting models (LightGBM and CatBoost) to predict the sale price of homes.

This project is intended as a learning exercise to understand real-world machine learning pipelines, model evaluation, and Kaggle competition workflow.

---

## Features

* **Data preprocessing and feature engineering**:

  * Handling missing values and categorical variables
  * Creating new features such as total square footage, house age, bathroom ratios, and presence of pool/garage/basement/fireplace
  * Skewness correction using log transformation for numeric features

* **Modeling**:

  * LightGBM regression with stratified k-fold cross-validation
  * CatBoost regression with CPU-friendly configuration
  * Optional blending of LightGBM and CatBoost predictions with automatic weight search

* **Evaluation**:

  * RMSE on log-transformed sale prices
  * Out-of-fold predictions for robust validation
  * Progress plotting of model training

* **Flexible configuration**:

  * Custom number of folds, random seed, and model choice
  * Ability to generate multiple submissions for averaging
  * Plotting training progress

---

## Results

* Kaggle Public Score: 13195.29291 RMSE
* Ranking: 112 / 4597 (~Top 2.44%)
* Key Techniques: CatBoost, Multi-seed averaging, Feature engineering

*Note: Scores and ranking are based on data from March 12, 2026.*

---

## Requirements

Python 3.8+ with the following libraries:

```
numpy
pandas
scikit-learn
lightgbm
catboost
matplotlib
tqdm
```

Install via pip:

```bash
pip install -r requirements.txt
```

---

## Usage

### Basic Command

```bash
python house_prices_boosting3.2.py --data_dir ./raw_data --out_dir ./output
```

### Using CatBoost only (recommended)

```bash
python house_prices_boosting3.2.py --data_dir ./raw_data --out_dir ./output --use_catboost true
```

### Enable Blending (LightGBM + CatBoost)

```bash
python house_prices_boosting3.2.py --data_dir ./raw_data --out_dir ./output --use_catboost true --blend true
```

### Multi-seed Average Submission

```bash
python house_prices_boosting3.2.py --data_dir ./raw_data --out_dir ./output --folds 5 --seed 42 --use_catboost true --plot true
python house_prices_boosting3.2.py --data_dir ./raw_data --out_dir ./output --folds 5 --seed 2024 --use_catboost true --plot true
cd ./output
python submission_average.py
```

### Parameters

* `--folds` : Number of CV folds (default 3, final 5)
* `--seed` : Random seed
* `--use_catboost` : Use CatBoost instead of LightGBM
* `--blend` : Blend CatBoost and LightGBM predictions
* `--auto_blend` : Automatically search for the best blend weight
* `--plot` : Save training progress plots
* `--cb_threads` : Number of threads for CatBoost
* `--cb_verbose` : CatBoost verbose iteration output

---

## File Structure

```
kaggle-house-price-prediction-competition/
│
├─ house_prices_boosting3.2.py    # Main training and prediction script
├─ requirements.txt               # Python dependencies
├─ raw_data/
│   ├─ train.csv                  # Training data
│   ├─ test.csv                   # Test data
│   └─ sample_submission.csv      # Sample submission template
└─ output/                        # Generated submissions and plots
```

---

## Methods

### Preprocessing

* Remove extreme outliers in living area vs sale price
* Encode categorical features
* Impute missing numeric values with median
* Create new features: total square footage, total bathrooms, porch area, house age, remodel age, presence indicators, ratio features, etc.
* Apply log transformation to skewed numeric features

### Modeling

* **LightGBM**: gradient boosting with CPU-friendly parameters and early stopping
* **CatBoost**: gradient boosting for categorical features with automatic handling
* **Blending**: weighted combination of LGBM and CatBoost predictions to minimize RMSE

### Evaluation

* RMSE on log-transformed sale prices
* Cross-validation with stratified folds for stable performance

---

## Output

* `submission_seed_{seed}.csv`: final predictions for Kaggle submission
* `progress_seed_{seed}.png`: optional training progress plots
* Supports multi-seed averaging for improved leaderboard performance

---

## References

* Kaggle House Prices dataset: [https://www.kaggle.com/c/house-prices-advanced-regression-techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
* LightGBM: [https://lightgbm.readthedocs.io](https://lightgbm.readthedocs.io)
* CatBoost: [https://catboost.ai](https://catboost.ai)

---

## License

This project is for learning purposes. Please credit if reused.
