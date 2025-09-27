import json
from pathlib import Path
import joblib, pandas as pd, numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Read the dataset
DATA = Path("Datasets/Crop_Yield_Fertilizer.csv")
OUT = Path("models"); OUT.mkdir(exist_ok=True)

FEATURES = ["N","P","K","temperature","humidity","ph","rainfall"]
TARGET = "label"

# Simply ensure all missing columns are considered, to avoid errors along the line
df = pd.read_csv(DATA)
missing_cols = set(FEATURES+[TARGET]) - set(df.columns)
if missing_cols:
    raise ValueError(f"Dataset missing columns: {missing_cols}")

# coerce numerics; if parsing fails, become NaN just incase
for c in FEATURES:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df[TARGET] = df[TARGET].astype(str)

# Now here we make sure to split the data in such a way 80% is used for training
# while 20% is used for testing
X, y = df[FEATURES], df[TARGET]
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


num_pre = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),  # harmless for random forests but its a good practice incase model changes
])

# This is for numeric pre-processing
pre = ColumnTransformer([
    ("num", num_pre, FEATURES)
])

# RandomForest model with 400 trees (n_estimators=400)
# class_weight="balanced" to automatically adjust weights inversely
# n_jobs=-1 is to use all available CPU cores for parallel training
# - random_state=42 just gives a seed
clf = RandomForestClassifier(
    n_estimators=400,
    class_weight="balanced",  
    n_jobs=-1,
    random_state=42
)

# This is the full pipeline to run
pipe = Pipeline([
    ("pre", pre),
    ("clf", clf)
])

# Now we train the pipeline
pipe.fit(Xtr, ytr)


pred = pipe.predict(Xva)
acc  = accuracy_score(yva, pred)
f1m  = f1_score(yva, pred, average="macro")

# Accuracy we conclude after training
print(f"Recommend | Acc={acc:.3f}  MacroF1={f1m:.3f}")
print(classification_report(yva, pred))

# We use a confusion matrix just to 
cm = confusion_matrix(yva, pred, labels=pipe.classes_)
cm_df = pd.DataFrame(cm, index=pipe.classes_, columns=pipe.classes_)
print("\nConfusion matrix (rows=true, cols=pred):\n", cm_df)

# 5-fold CV for robustness
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
cv_f1  = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
print(f"\nCV Accuracy: {cv_acc.mean():.3f}±{cv_acc.std():.3f} | CV Macro-F1: {cv_f1.mean():.3f}±{cv_f1.std():.3f}")

#  save artifacts 
joblib.dump(pipe, OUT / "recommend_pipeline.joblib")
# save the model’s class order (don’t rely on sorted() later)
json.dump(list(map(str, pipe.classes_)), open(OUT / "recommend_classes.json","w"))
print("\nSaved models/recommend_pipeline.joblib and recommend_classes.json")
