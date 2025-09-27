from __future__ import annotations
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from classifierTest import callModelOne
# configuration
MODEL_B_PATH = Path("models/yield_pipeline.joblib")

NUM_FEATURES = ["N","P","K","temperature","humidity","ph","rainfall"]
CAT_FEATURES = ["fertilizer","label"]  # we'll set label per hypothesis
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES


def normalize_probs(pairs: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Ensure probabilities sum to 1 (if they don’t already)."""
    probs = np.array([p for _, p in pairs], dtype=float)
    s = probs.sum()
    if s <= 0:
        # fallback to uniform if something went wrong
        n = max(1, len(pairs))
        return [(c, 1.0 / n) for c, _ in pairs]
    probs = probs / s
    return [(pairs[i][0], float(probs[i])) for i in range(len(pairs))]


def run_model_b_for_hypotheses(
    model_b,
    base_features: Dict[str, Any],
    model_a_pairs: List[Tuple[str, float]],
    *,
    top_k: int | None = None,
    min_conf: float = 0.0,
    normalize: bool = True,
) -> Dict[str, Any]:
    """
    Parameters
    ----------
    model_b : fitted sklearn Pipeline (your RandomForest pipeline).
    base_features : dict with keys for ALL_FEATURES except 'label'. Example:
        {
          "N": 90, "P": 42, "K": 43,
          "temperature": 24.5, "humidity": 60, "ph": 6.5, "rainfall": 140.0,
          "fertilizer": "npk 20-20-20"
        }
    model_a_pairs : list of (crop_label, confidence).
    top_k : keep only top-k hypotheses (after filtering).
    min_conf : drop hypotheses with confidence below this.
    normalize : if True, renormalize confidences to sum to 1 after filtering.

    Returns
    -------
    dict with:
      - "hypotheses": list of dicts [{crop, prob, yield_pred}, ...]
      - "expected_yield": float (sum_i prob_i * yield_pred_i)
      - "meta": misc info
    """
    # Filter by min_conf
    filtered = [(c, p) for (c, p) in model_a_pairs if p >= min_conf]
    # Sort desc by confidence
    filtered.sort(key=lambda x: x[1], reverse=True)
    # Keep top_k if specified
    if top_k is not None:
        filtered = filtered[:top_k]
    # Normalize if requested
    pairs = normalize_probs(filtered) if normalize else filtered

    # Prepare rows for each hypothesis (one row per crop label)
    rows = []
    crops = []
    probs = []
    for crop_label, prob in pairs:
        # build one row dict for Model B
        row = dict(base_features)  # copy
        row["label"] = str(crop_label).strip().lower()
        # ensure fertilizer is normalized like in training
        if "fertilizer" in row and row["fertilizer"] is not None:
            row["fertilizer"] = str(row["fertilizer"]).strip().lower()
        rows.append(row)
        crops.append(crop_label)
        probs.append(prob)

    if not rows:
        return {
            "hypotheses": [],
            "expected_yield": None,
            "meta": {"note": "No hypotheses passed filters."}
        }

    X = pd.DataFrame(rows, columns=ALL_FEATURES)

    # Predict with Model B
    y_pred = model_b.predict(X)  # shape (H,)
    y_pred = y_pred.astype(float)

    hypotheses = []
    for c, p, yp in zip(crops, probs, y_pred):
        hypotheses.append({
            "crop": c,
            "prob": float(p),
            "yield_pred": float(yp)
        })

    expected_yield = float(np.dot(np.array(probs, dtype=float), y_pred))

    return {
        "hypotheses": hypotheses,
        "expected_yield": expected_yield,
        "meta": {
            "normalized": normalize,
            "top_k": top_k,
            "min_conf": min_conf,
            "num_hypotheses": len(hypotheses)
        }
    }


def main(Nitrogen, Phosphorus, Potassium, pH, fertilizer, Temperature, Humidity, rainfall):
    # ---- Load Model B ----
    model_b = joblib.load(MODEL_B_PATH)

    # ---- Example inputs (replace with your real runtime inputs) ----
    # Model A output (crop, confidence). E.g., from your classifier:
    # model_a_pairs = [
    #     ("wheat", 0.62),
    #     ("rice", 0.28),
    #     ("maize", 0.10),
    # ]


    
    model_a_pairs = callModelOne(Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, rainfall)

    # Base features for this plot/sample (everything except label)
    base_features = {
        "N": Nitrogen, "P": Phosphorus, "K": Potassium,
        "temperature": Temperature, "humidity": Humidity, "ph": pH, "rainfall": rainfall,
        "fertilizer": fertilizer
    }

    results = run_model_b_for_hypotheses(
        model_b=model_b,
        base_features=base_features,
        model_a_pairs=model_a_pairs,
        top_k=3,
        min_conf=0.02,
        normalize=True
    )

    final_output = []
    for h in results["hypotheses"]:
        final_output.append({
            "crop": h["crop"].capitalize(),        # Crop name
            "yield": round(h["yield_pred"], 2),    # Yield prediction
            "confidence": round(h["prob"], 2)      # Probability / confidence
        })
        
    print(final_output)
        
    # Pretty print + save
    print(json.dumps(results, indent=2))
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/ensemble_result.json", "w") as f:
        json.dump(results, f, indent=2)

    # Also tabular CSV of hypotheses
    if results["hypotheses"]:
        df = pd.DataFrame(results["hypotheses"])
        df.to_csv("outputs/ensemble_hypotheses.csv", index=False)
        print("Wrote outputs/ensemble_hypotheses.csv")
        
    return final_output


if __name__ == "__main__":
    main()