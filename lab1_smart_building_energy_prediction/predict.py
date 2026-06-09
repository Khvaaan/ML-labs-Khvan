from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path("models/best_energy_model.pkl")
FEATURES_PATH = Path("models/feature_columns.pkl")


def main():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)

    sample = {
        "hour": 14,
        "day_of_week": 2,
        "is_weekend": 0,
        "outside_temperature": 28.5,
        "humidity": 52.0,
        "occupancy_count": 110,
        "lighting_usage": 0.72,
        "equipment_load_kw": 39.0,
        "hvac_load_kw": 31.0,
        "co2_ppm": 980.0,
        "active_rooms": 12,
    }

    X = pd.DataFrame([sample])[feature_columns]
    prediction = model.predict(X)[0]
    print("Input data:")
    for key, value in sample.items():
        print(f"{key}: {value}")
    print(f"\nPredicted energy consumption: {prediction:.2f} kWh")

if __name__ == "__main__":
    main()
