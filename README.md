## Agriculture Crop Recommendation System
As agriculture is a huge part of our economy in the MENA region, we decided to build a system that helps farmers make accurate agronomic decisions. 
The app predicts suitable crops for a given location depending on input information and local environmental conditions.

## Data set used: crop yield fertilizer https://huggingface.co/datasets/Jakehills/Crop_Yield_Fertilizer/viewer/default/train?p=249 

## API
### interactive Streamlit that recommends crops and shows expected yield for a selected city/quarter in the MENA region, combining location-based weather, soil inputs (N, P, K, pH), and fertilizer choice.
### User selects a city in the MENA region; coordinates (lon,lat) are then geocoded in the backend using OpenStreetMap (OSM) Nominatim. Openmeteo Archive API in turn fetches seasonal temperature, humidity, and rainfall of that area. We also quarter map the year into 4 quarters of 3 months. Those features are then merged with user input N,P,K and fertilizer details.
### The model ranks crops from most to least likely and shows the ones above a chosen probability threshold, and returns the options above a configurable probability threshold, so users can see the top recommendations alongside their confidence scores. Seasonality controls let users explore what time to plant when for better timing decisions.
### Website displays a weather summary (cards for temperature, humidity, rainfall), a map pin of the selected location, and crop recommendations. 

### training 
### Model A: Classification: the data is split into 80% training and 20% testing. This stratification preserves class proportions across training and validation, which stabilizes metrics on imbalanced data. No leakage occurs as preprocessing and modeling are wrapped in a pipeline, so statistics are learned only on the training fold. We use random forest classfiier, ensemble that handles nonlinear interactions. 
### Model B: Regression: Predicts crop yield from soil, weather, crop label, and fertilizer information. The training pipeline is end-to-end: numeric features undergo median imputation → standardization, while categorical features use most-frequent imputation → one-hot encoding (safely ignoring unseen categories at inference). A Random Forest Regressor provides stable performance. All steps live in a single scikit-learn Pipeline, so preprocessing learned on the training data is applied identically at inference—preventing data leakage. We use an 80/20 hold-out split for evaluation and report MAE, RMSE, and R² to capture absolute error, error scale, and explained variance. To assess robustness, we run shuffled 5-fold K-Fold cross-validation and summarize results as mean ± standard deviation for MAE, RMSE, and R², indicating stability across splits.
### Model A : Which crops are suitable here? → gives a probability for each crop.
### Model B : If we plant crop X here, how much yield can we expect? → gives a numeric yield.
### Combining them lets us show both likelihood (suitability) and magnitude (expected yield), and compute an expected outcome
### Training on main: This component fuses the two models’ strengths to produce decision-ready outputs. It takes crop candidates and confidences from Model A, evaluates each candidate’s predicted yield via Model B under the same field conditions, and returns both a ranked hypothesis list and a single expected yield summary. 
### WeatherAPI: This FastAPI service provides the environmental conditions needed by the Crop Recommendation System for a user-selected location. It aggregates weather (temperature, humidity, precipitation) and soil pH from free public APIs and returns a compact JSON payload the models can consume.


