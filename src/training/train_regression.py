# src/train_yield.py
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# configuration
DATA = Path("Datasets/Crop_Yield_Fertilizer.csv")
OUT  = Path("models"); OUT.mkdir(exist_ok=True)

NUM_FEATURES = ["N","P","K","temperature","humidity","ph","rainfall"]
CAT_FEATURES = ["label","fertilizer"]   # keep fertilizer as string category
TARGET       = "yield"
MODEL_PATH   = OUT / "yield_pipeline.joblib"
META_PATH    = OUT / "yield_meta.json"


def load_and_clean(path: Path):
    df = pd.read_csv(path)

    # validating schema
    needed = set(NUM_FEATURES + CAT_FEATURES + [TARGET])
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Converting numeric fields; invalid entries become NaN
    for c in NUM_FEATURES + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ncategoricals are normalized
    for c in CAT_FEATURES:
        df[c] = (df[c].astype(str)
                       .str.strip()
                       .str.lower())

    # drop rows of no target
    df = df.dropna(subset=[TARGET])

    return df


def build_pipeline():
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, NUM_FEATURES),
        ("cat", cat_pipe, CAT_FEATURES),
    ])

    reg = RandomForestRegressor(
        n_estimators=600,
        min_samples_leaf=2,   
        n_jobs=-1,
        random_state=42
    )

    pipe = Pipeline([
        ("pre", pre),
        ("reg", reg),
    ])
    return pipe


def main():
    df = load_and_clean(DATA)

    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df[TARGET]

    
    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = build_pipeline()
    pipe.fit(Xtr, ytr)

    
    pred = pipe.predict(Xva)
    mae  = mean_absolute_error(yva, pred)
    mse  = mean_squared_error(yva, pred)
    rmse = mse ** 0.5
    r2   = r2_score(yva, pred)
    print(f"Yield | MAE={mae:,.3f}  RMSE={rmse:,.3f}  R²={r2:.3f}")

    # 5-fold CV (mean±std)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mae  = -cross_val_score(pipe, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
    cv_rmse = -cross_val_score(pipe, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    cv_r2   =  cross_val_score(pipe, X, y, cv=cv, scoring="r2", n_jobs=-1)
    print(f"CV   | MAE={cv_mae.mean():,.3f}±{cv_mae.std():.3f}  "
          f"RMSE={cv_rmse.mean():,.3f}±{cv_rmse.std():.3f}  "
          f"R²={cv_r2.mean():.3f}±{cv_r2.std():.3f}")

    # save model + meta
    joblib.dump(pipe, MODEL_PATH)
    json.dump({
        "num_features": NUM_FEATURES,
        "cat_features": CAT_FEATURES,
        "target": TARGET
    }, open(META_PATH, "w"))

    print(f"Saved {MODEL_PATH.name} and {META_PATH.name} in {OUT.resolve()}")


if __name__ == "__main__":
    main()