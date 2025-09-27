import json, joblib, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

DATA = Path("Datasets/Crop_recommendation.csv")
OUT_DIR = Path("models"); OUT_DIR.mkdir(exist_ok=True)

FEATURES = ["N","P","K","temperature","humidity","ph","rainfall"]
TARGET   = "label"

df = pd.read_csv(DATA)

# (Why split stratified?) Keeps rare crops present in both sets → fair evaluation.
X, y = df[FEATURES], df[TARGET]
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# (Why pipeline?) Ensures identical transforms at train + inference.
pipe = Pipeline([
    ("scaler", StandardScaler()),                # RF doesn’t need it, but it’s harmless and future-proofs
    ("clf", RandomForestClassifier(
        n_estimators=400, n_jobs=-1, random_state=42
    ))
])
pipe.fit(Xtr, ytr)

pred = pipe.predict(Xva)
acc  = accuracy_score(yva, pred)
f1m  = f1_score(yva, pred, average="macro")
print(f"Recommend | Acc={acc:.3f}  MacroF1={f1m:.3f}")
print(classification_report(yva, pred))

joblib.dump(pipe, OUT_DIR / "recommend_pipeline.joblib")
json.dump(sorted(y.unique().tolist()), open(OUT_DIR/"recommend_classes.json","w"))
print("Saved models/recommend_pipeline.joblib")