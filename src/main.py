import joblib
import numpy as np
import pandas as pd

# load the trained pipeline

# FEATURES = ["N","P","K","temperature","humidity","ph","rainfall"]
pipe = joblib.load("models/recommend_pipeline.joblib")

# one row of input features
Nitrogen = 0
Phosphorus = 0
Potassium = 0
Temperature = 0
Humidity = 0
pH = 0
rainfall = 0

X_new = np.array([[Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, rainfall]])

data = pd.DataFrame([{
    "N": 90, "P": 40, "K": 40,
    "temperature": 25.0,
    "humidity": 80.0,
    "ph": 6.5,
    "rainfall": 200.0
}])

# # get prediction
# crop = pipe.predict(data)[0]
# print("Recommended crop:", crop)

# predict probabilities for all crops
probs = pipe.predict_proba(data)[0]   # 1D array with probability per class
classes = pipe.classes_              # list of crop labels (in the same order)

# pair them and sort by probability
results = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)

def pick_matches(classes, probs, p_min=0.5, min_k=3):
    pairs = sorted(zip(classes, probs), key=lambda t: t[1], reverse=True)
        # only ≥ 0.5
    print("all pairs:", pairs)  
    # keep those above threshold
    keep = [p for p in pairs if p[1] >= p_min]
    
    return keep

# usage:
matches = pick_matches(classes, probs, p_min=0.5, min_k=3)
print("filtered:", results) 
for crop, p in matches:
    print(f"{crop:12s} {p*100:.2f}%")


