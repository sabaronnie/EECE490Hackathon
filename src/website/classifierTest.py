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

#X_new = np.array([[Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, rainfall]])

def callModelOne(Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, rainfall):

    data = pd.DataFrame([{
        "N": Nitrogen, "P": Phosphorus, "K": Potassium,
        "temperature": Temperature,
        "humidity": Humidity,
        "ph": pH,
        "rainfall": rainfall
    }])


    # predict probabilities for all crops
    probs = pipe.predict_proba(data)[0]   # 1D array with probability per class
    classes = pipe.classes_              # list of crop labels (in the same order)

    # pair them and sort by probability
    results = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)

    pairs = sorted(zip(classes, probs), key=lambda t: t[1], reverse=True)
        # only ≥ 0.5
    print("all pairs:", pairs)  
    # keep those above threshold

    p_min = 0.15
    # usage:
    matches = [p for p in pairs if p[1] >= p_min]

    output = []
    for crop, p in matches:
        output.append((crop, p))
        
    print("\nRecommended crops (threshold = {:.0%}):".format(p_min))
    for crop, p in matches:
        print(f"  - {crop:<12} {p*100:.2f}%")
    
    return output


            
            
            


