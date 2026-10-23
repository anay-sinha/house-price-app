# train.py
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer, make_column_selector as sel
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from features import add_features

df = add_features(pd.read_csv("data/train.csv"))
df = df[df["GrLivArea"] < 4500]                    # drop outliers
y = np.log1p(df.pop("SalePrice")); X = df.drop(columns=["Id"])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), sel(dtype_include="number")),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="None")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))]), sel(dtype_exclude="number")),
])

models = {
    "Ridge": Ridge(alpha=10),
    "RandomForest": RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=4,
                            subsample=0.8, colsample_bytree=0.8, random_state=42),
}

for name, m in models.items():
    pipe = Pipeline([("pre", pre), ("model", m)])
    cv = -cross_val_score(pipe, X_tr, y_tr, cv=5,
                          scoring="neg_root_mean_squared_error").mean()
    print(f"{name}: CV RMSE (log) = {cv:.4f}")
