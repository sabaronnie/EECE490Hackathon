import requests
import datetime
import pandas as pd

MENA_COUNTRIES = [
    "Lebanon", "Jordan", "Syria", "Iraq", "Egypt",
    "Saudi Arabia", "United Arab Emirates", "Qatar", "Kuwait", "Bahrain", "Oman", "Yemen",
    "Morocco", "Algeria", "Tunisia", "Libya", "Sudan"
]

FERTILIZERS = ["DAP", "Urea","Gypsum", "Lime", "MOP", "Rhizobium", "SSP", "Potassium Nitrate", "Rock Phosphate"]

MENA_CITIES = {
    "Lebanon": ["Beirut", "Tripoli", "Zahle", "Sidon", "Byblos", "Tyre", "Beqaa","Nabatieh", "Aley"],
    "Jordan": ["Amman", "Irbid", "Zarqa", "Madaba", "Aqaba", "Mafrak"],
    "Syria": ["Damascus", "Aleppo", "Homs", "Latakia", "Hama", "Tartus"],
    "Iraq": ["Baghdad", "Basra", "Mosul", "Erbil", "Najaf"],
    "Egypt": ["Cairo", "Alexandria", "Giza", "Luxor", "Aswan", "Sharm El-Sheikh"],
    "Saudi Arabia": ["Riyadh", "Jeddah", "Maqqa", "Medina", "Dammam", "Al Khobar"],
    "United Arab Emirates": ["Dubai", "Abu Dhabi", "Sharjah", "Al Ain", "Fujairah"],
    "Qatar": ["Doha", "Al Rayyan", "Al Wakrah"],
    "Kuwait": ["Kuwait City", "Hawalli", "Al Ahmadi"],
    "Bahrain": ["Manama", "Muharraq", "Riffa"],
    "Oman": ["Muscat", "Salalah", "Sohar", "Nizwa"],
    "Yemen": ["Sana'a", "Aden", "Taiz", "Al Hudaydah"],
    "Morocco": ["Casablanca", "Rabat", "Marrakech", "Fes", "Tangier", "Agadir"],
    "Algeria": ["Algiers", "Oran", "Constantine", "Annaba", "Blida"],
    "Tunisia": ["Tunis", "Sfax", "Sousse", "Bizerte", "Gabes"],
    "Libya": ["Tripoli", "Benghazi", "Misrata", "Sabha"],
    "Sudan": ["Khartoum", "Omdurman", "Port Sudan", "Kassala"]
}



def geocode_location(location_name: str):
    """
    Use OpenStreetMap Nominatim API to convert place name into lat/lon.
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": location_name,
        "format": "json",
        "limit": 1
    }

    response = requests.get(url, params=params, headers={"User-Agent": "streamlit-app"})
    data = response.json()

    if data:
        lat = float(data[0]["lat"])
        lon = float(data[0]["lon"])
        return lat, lon
    else:
        return None, None
    

def get_quarter_weather(lat, lon, year, quarter):
    """
    Fetch daily weather data and compute averages for the given quarter.
    """


    # Quarter → start and end dates
    quarters = {
        "Q1": ("01-01", "03-31"),
        "Q2": ("04-01", "06-30"),
        "Q3": ("07-01", "09-30"),
        "Q4": ("10-01", "12-31")
    }
    start = f"{year}-{quarters[quarter][0]}"
    end = f"{year}-{quarters[quarter][1]}"

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "relative_humidity_2m_mean"],
        "timezone": "auto"
    }
    r = requests.get(url, params=params)
    data = r.json()

    # Convert to DataFrame
    df = pd.DataFrame({
        "date": data["daily"]["time"],
        "tmax": data["daily"]["temperature_2m_max"],
        "tmin": data["daily"]["temperature_2m_min"],
        "rain": data["daily"]["precipitation_sum"],
        "humidity": data["daily"]["relative_humidity_2m_mean"]
    })

    # Compute averages
    avg_temp = (df["tmax"].mean() + df["tmin"].mean()) / 2
    avg_humidity = df["humidity"].mean()
    avg_rainfall = df["rain"].mean()

    return avg_temp, avg_humidity, avg_rainfall
