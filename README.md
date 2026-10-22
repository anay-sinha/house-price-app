# 🏠 House Price Predictor

An end-to-end machine learning project: clean a real-world tabular dataset, engineer features, train and compare models, and serve predictions through an interactive **Streamlit** web app.

**Repository:** https://github.com/anay-sinha/house-price-app
**Live demo:** _coming soon (Streamlit Community Cloud)_

<!-- After deploying, add a screenshot: save it as assets/screenshot.png and uncomment the line below -->
<!-- ![App screenshot](assets/screenshot.png) -->

---

## Overview

Given a handful of house characteristics (overall quality, total area, bathrooms, garage size, age, neighborhood), the app estimates the sale price in real time as you move the sliders.

- **Dataset:** [House Prices: Advanced Regression Techniques (Ames, Iowa)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
- **Target:** `SalePrice`, modeled as `log1p(SalePrice)` because the raw target is right-skewed
- **Models compared:** Ridge, Random Forest, XGBoost
- **Metric:** RMSE on the log scale (RMSLE), plus RMSE in dollars

## Results

| Model         | CV RMSE (log) | Test RMSE (log) |
|---------------|---------------|-----------------|
| Ridge         | _fill in_     | _fill in_       |
| Random Forest | _fill in_     | _fill in_       |
| XGBoost       | _fill in_     | _fill in_       |

_The slim model used in the app (top features only) scores: **fill in**. The test set was evaluated once, after tuning._

## Project structure

```
house-price-app/
├── data/
│   └── .gitkeep           # placeholder; train.csv goes here (see Setup)
├── notebooks/
│   └── eda.ipynb          # exploratory data analysis
├── .gitignore
├── app.py                 # Streamlit web app (loads the saved model only)
├── features.py            # shared feature engineering (training + app)
├── train.py               # pipeline, model comparison, tuning, saves model.joblib
├── model.joblib           # trained slim pipeline + neighborhood list
├── requirements.txt
└── README.md
```

## Methodology

1. **EDA:** checked missing values, target skew, correlations and outliers (a few very large, cheaply sold houses were dropped: `GrLivArea >= 4500`).
2. **Missing values:** in columns like `PoolQC`, `Alley` and `GarageType`, NaN means the feature is absent and is filled with `"None"`. True numeric gaps use the median. Imputation happens inside a scikit-learn `Pipeline` to prevent data leakage.
3. **Categorical encoding:** one-hot encoding with `handle_unknown="ignore"`.
4. **Feature engineering** (`features.py`): `TotalSF`, `HouseAge`, `RemodAge`, `TotalBath`, `HasGarage`, `QualxSF`.
5. **Why not price per square foot?** It is computed from the target (`SalePrice / sqft`), so using it as an input would leak the answer. It is used for EDA only.
6. **Modeling:** Ridge, Random Forest and XGBoost compared with 5-fold cross-validation; the best model was tuned with `RandomizedSearchCV`.
7. **Slim model:** a pipeline retrained on the top features by importance so the app form stays usable.

## Setup

**Requirements:** Python 3.10+

```bash
# 1. Clone the repo
git clone https://github.com/anay-sinha/house-price-app.git
cd house-price-app

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate           # Windows
source venv/bin/activate        # macOS / Linux

# 3. Install dependencies
python -m pip install -r requirements.txt
```

**Dataset:** `train.csv` is not included in this repository. Download it from the [Kaggle competition page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) and place it at `data/train.csv`. It is only needed if you want to retrain the model; the app runs from the included `model.joblib`.

## Usage

**Run the app** (uses the saved `model.joblib`):

```bash
streamlit run app.py
```

Then open http://localhost:8501.

**Retrain the model** (requires `data/train.csv`; regenerates `model.joblib`):

```bash
python train.py
```

## Requirements

Main libraries (see `requirements.txt`):

- pandas, numpy
- scikit-learn
- xgboost
- joblib
- streamlit
- matplotlib, seaborn (EDA notebook)

`model.joblib` must be loaded with the same scikit-learn and XGBoost versions it was trained with, so keep the versions in `requirements.txt` pinned.

## Deployment

Deploy on [Streamlit Community Cloud](https://share.streamlit.io): sign in with GitHub, select `anay-sinha/house-price-app`, set the main file to `app.py`, and deploy. The app only loads `model.joblib`; it never retrains.

## Limitations

- Trained on Ames, Iowa sales from 2006–2010, so it will not generalize to other cities or current prices.
- The app uses a reduced feature set; predictions are estimates, not appraisals.
- Error figures are on the held-out test set from one train/test split.

## Future work

- SHAP explanations for individual predictions
- Price-vs-area curve in the app
- Extend the pipeline to a loan default classifier (PR-AUC, threshold tuning)

## License

Add a `LICENSE` file (e.g. MIT) to the repo. The dataset is subject to Kaggle's and the Ames Housing Dataset's own terms.

## Author

[anay-sinha](https://github.com/anay-sinha)
