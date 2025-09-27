# backendd.py
import numpy as np
import httpx
from typing import Optional
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import asyncio

app = FastAPI(title="Conditions API (No Google)")

# External free APIs
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
SOILGRIDS = "https://rest.isric.org/soilgrids/v2.0/properties/query"

async def fetch_weather(lat: float, lon: float):
    """Return representative values: temperature (°C), humidity (%), rain (mm, 14-day total)."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum",
        "timezone": "auto",
        "forecast_days": 175
    }
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(OPEN_METEO, params=params)
        r.raise_for_status()
        d = r.json()["daily"]

    temperature = float(np.nanmean(d["temperature_2m_mean"]))        # °C
    humidity    = float(np.nanmean(d["relative_humidity_2m_mean"]))  # %
    rain_175d    = float(np.nansum(d["precipitation_sum"]))           

    return {
        "temperature": round(temperature, 2),
        "humidity": round(humidity, 1),
        "rain": round(rain_175d, 1)
    }

async def fetch_soil_ph(lat: float, lon: float) -> Optional[float]:
    """Soil pH(H2O) at 0–5 cm. SoilGrids stores pH*10 → divide by 10."""
    params = {"lat": lat, "lon": lon, "property": "phh2o", "depth": "0-5cm"}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(SOILGRIDS, params=params)
            if r.status_code != 200:
                return None
            data = r.json()
        ph10 = data["properties"]["layers"][0]["depths"][0]["values"]["M"]  # median
        return round(float(ph10) / 10.0, 2)
    except Exception:
        return None

@app.get("/api/conditions")
async def conditions(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    include_ph: bool = Query(True)
):
    """Return exactly: temperature (°C), humidity (%), rain (mm, 14-day total), ph (0–14 or null)."""
    try:
        wx = await fetch_weather(lat, lon)
    except Exception as e:
        return JSONResponse({"error": f"weather fetch failed: {e}"}, status_code=502)

    result = dict(wx)
    result["ph"] = await fetch_soil_ph(lat, lon) if include_ph else None
    return result

if __name__ == "__main__":
    result = asyncio.run(fetch_weather(33.8886, 35.4955))
    
    print(result)
    print(asyncio.run(fetch_soil_ph(33.8886, 35.4955)))