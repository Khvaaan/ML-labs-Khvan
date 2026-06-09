import math
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def main():
    rng = np.random.default_rng(42)
    n = 2400
    start = datetime(2026, 1, 1)
    rows = []

    for i in range(n):
        dt = start + timedelta(hours=i)
        hour = dt.hour
        day_of_week = dt.weekday()
        is_weekend = int(day_of_week >= 5)
        season_angle = 2 * math.pi * (dt.timetuple().tm_yday / 365)

        outside_temperature = 7 + 16 * np.sin(season_angle - 0.8) + rng.normal(0, 3.5)
        humidity = np.clip(58 - 0.7 * outside_temperature + rng.normal(0, 8), 25, 95)

        working_time = (8 <= hour <= 19) and not is_weekend
        base_occupancy = 95 if working_time else (25 if 10 <= hour <= 22 else 6)
        occupancy_count = int(max(0, rng.normal(base_occupancy, 18 if working_time else 8)))

        lighting_usage = np.clip((1 if (hour < 8 or hour > 17) else 0.45) + occupancy_count / 260 + rng.normal(0, 0.08), 0.05, 1.0)
        equipment_load_kw = np.clip(8 + occupancy_count * 0.18 + (12 if working_time else 3) + rng.normal(0, 3), 4, 60)
        hvac_load_kw = np.clip(abs(outside_temperature - 20) * 2.1 + occupancy_count * 0.035 + rng.normal(0, 2), 0, 55)
        co2_ppm = np.clip(420 + occupancy_count * 5.2 + rng.normal(0, 40), 380, 1600)
        active_rooms = int(np.clip(round(2 + occupancy_count / 12 + rng.normal(0, 2)), 1, 24))

        energy_consumption_kwh = (
            18 + 0.95 * equipment_load_kw + 1.15 * hvac_load_kw + 17 * lighting_usage
            + 0.08 * occupancy_count + 0.012 * co2_ppm + (4 if working_time else -2)
            + rng.normal(0, 5)
        )

        rows.append({
            "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "outside_temperature": round(outside_temperature, 2),
            "humidity": round(humidity, 2),
            "occupancy_count": occupancy_count,
            "lighting_usage": round(lighting_usage, 3),
            "equipment_load_kw": round(equipment_load_kw, 2),
            "hvac_load_kw": round(hvac_load_kw, 2),
            "co2_ppm": round(co2_ppm, 1),
            "active_rooms": active_rooms,
            "energy_consumption_kwh": round(max(5, energy_consumption_kwh), 2),
        })

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "building_energy.csv", index=False)
    print("Saved data/building_energy.csv", df.shape)

if __name__ == "__main__":
    main()
