# streamlit_app.py

from utils import geocode_location, get_quarter_weather, MENA_COUNTRIES, MENA_CITIES, FERTILIZERS
import streamlit as st
import pandas as pd
import numpy as np
import joblib  # to load ML models
import datetime
import time
# from src.website.main import run_model_b_for_hypotheses
# from src.website.classifierTest import callModelOne

from main import main
# Load your pre-trained models
# crop_model = joblib.load("crop_model.pkl")
# yield_model = joblib.load("yield_model.pkl")

st.title("🌾 Smart Farmer: Crop & Yield Recommendation")
st.write("Get AI-powered crop suggestions and yield forecasts based on soil & weather data.")

# --- Input Form ---
st.header("Enter Farm Details")
col1, col2 = st.columns(2)

year = datetime.date.today().year - 1  # last complete year

with col1:
    nitrogen = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
    phosphorus = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=40)
    potassium = st.number_input("Potassium (K)", min_value=0, max_value=200, value=40)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)

with col2:
    fertilizer = st.selectbox("Fertilizer Type", FERTILIZERS)
    quarter = st.selectbox("Select Quarter", ["Q1", "Q2", "Q3", "Q4"])
    country = st.selectbox("Select Country", MENA_COUNTRIES)
    city = st.selectbox("Select City/Region", MENA_CITIES[country])
    # state = st.text_input("State / Region", "Lebanon")

# Get weather averages
location = f"{city}, {country}"
lat, lon = geocode_location(location)
avg_temp, avg_humidity, avg_rainfall = get_quarter_weather(lat, lon, year, quarter)  


# --- Prediction Button ---
if st.button("Recommend Crops & Predict Yield"):
    recommendations = main(nitrogen, phosphorus, potassium, ph, fertilizer, avg_temp, avg_humidity, avg_rainfall*90)
    steps = [
        "🔍 Fetching weather data...",
        "🧪 Analyzing soil nutrients...",
        "🤖 Running AI crop model...",
        "📊 Generating yield forecast..."
    ]
    progress = st.progress(0)
    status = st.empty()

    for i, step in enumerate(steps):
        status.text(step)
        progress.progress(int((i+1)/len(steps)*100))
        time.sleep(1.5)  # simulate processing time

    progress.empty()
    status.empty()


    st.success("✅ Done! Recommendations ready 👇")

    

    # Display weather data
    st.subheader(f"🌤️ Weather Summary for {quarter}")

    weather_data = [
        {"label": "Avg Temp", "value": f"{avg_temp:.2f} °C", "emoji": "🌡️", "color": "#FFB347"},
        {"label": "Avg Humidity", "value": f"{avg_humidity:.2f} %", "emoji": "💧", "color": "#77B5FE"},
        {"label": "Avg Rainfall", "value": f"{avg_rainfall:.2f} mm/day", "emoji": "🌧️", "color": "#90EE90"}
    ]

    cols = st.columns(len(weather_data))
    for i, wd in enumerate(weather_data):
        with cols[i]:
            with st.container(border=True):
                st.markdown(
                    f"<div style='text-align: center;'>"
                    f"<h3>{wd['emoji']} {wd['label']}</h3>"
                    f"<p style='font-size: 20px; font-weight: bold; color: {wd['color']};'>{wd['value']}</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )



    # {"crop": "Rice", "yield": 72.5, "confidence": 0.82},
    # {"crop": "Wheat", "yield": 65.3, "confidence": 0.74},
    # {"crop": "Maize", "yield": 80.1, "confidence": 0.91}
    # ]

    # Display Crop Recommendations
    st.subheader(f"🌱 Recommended Crops for {quarter}")

    cols = st.columns(len(recommendations))
    for i, rec in enumerate(recommendations):
        with cols[i]:
            with st.container(border=True):
                st.markdown(f"### {rec['crop']}")
                st.metric("Expected Yield (kg/ha)", f"{rec['yield']:.3f}")
                st.progress(rec['confidence'])

    # Show farm location on map
    df_map = pd.DataFrame({"lat": [lat], "lon": [lon]})

    st.subheader(f"📍 Location: {city}, {country}")
    st.map(df_map, zoom=12, use_container_width=True)