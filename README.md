# House Prices Prediction

A machine learning project to predict house sale prices based on features like lot area, neighborhood, and overall house condition/quality, using the [Kaggle "House Prices: Advanced Regression Techniques"](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) dataset.

## Approach

- **Exploratory data analysis**: Examined relationships between features (e.g. total square footage, overall quality) and `SalePrice`, including scatterplots, boxplots, and a correlation heatmap of the top features.
- **Feature engineering**:
  - Filled missing values in basement/floor square footage columns (`TotalBsmtSF`, `1stFlrSF`, `2ndFlrSF`) with 0.
  - Created a new `TotalSF` feature (basement + 1st floor + 2nd floor square footage).
  - Applied a log transformation (`np.log1p`) to `TotalSF` and `SalePrice` to reduce skewness.
  - Removed outliers in `TotalSF` using the IQR method.
- **Preprocessing pipeline**: Built a `scikit-learn` pipeline with separate handling for numerical features (mean imputation + standardization) and categorical features (imputation + one-hot encoding).
- **Modeling**:
  - **Linear Regression** baseline — RMSE ≈ 34,710.89, R² ≈ 78.5%
  - **XGBoost** with grid search hyperparameter tuning — outperformed linear regression on both RMSE and R², showing the benefit of a more complex model for capturing non-linear relationships in the data.

## Repo contents

| File | Description |
| --- | --- |
| `House Prices - Advanced Regression Techniques.ipynb` | Main notebook: EDA, feature engineering, preprocessing, and model training (Linear Regression vs. XGBoost) |
| `submission.ipynb` | Notebook for generating the final Kaggle submission |
| `train.csv` | Training data with house features and `SalePrice` |
| `test.csv` | Test data for generating predictions |
| `data_description.txt` | Description of all dataset columns |
| `sample_submission.csv` | Kaggle sample submission format |
| `submission.csv` | Final predictions submitted to Kaggle |

## Requirements

- Python 3 with `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`
- Jupyter Notebook

## Running it

1. Clone the repo and open `House Prices - Advanced Regression Techniques.ipynb` in Jupyter.
2. Run the cells in order: data loading → EDA → feature engineering → preprocessing pipeline → model training/evaluation.
3. Use `submission.ipynb` to generate predictions on `test.csv` in the format of `sample_submission.csv`.

## Data source

Dataset from the Kaggle competition [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).
