from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path("models/best_anomaly_model.pkl")
SCALER_PATH = Path("models/scaler.pkl")
FEATURES_PATH = Path("models/feature_columns.pkl")


def main():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURES_PATH)

    sample = {
        "temperature": 23.5,
        "humidity": 44.0,
        "co2_ppm": 920.0,
        "motion_level": 0.70,
        "power_kw": 62.0,
        "occupancy_count": 105,
        "vibration_level": 0.14,
        "network_latency_ms": 35.0,
        "packet_loss_percent": 0.25,
        "battery_level": 82.0,
    }

    X = pd.DataFrame([sample])[feature_columns]
    X_scaled = scaler.transform(X)
    raw_prediction = model.predict(X_scaled)[0]
    status = "anomaly" if raw_prediction == -1 else "normal"

    print("Input sensor state:")
    for key, value in sample.items():
        print(f"{key}: {value}")
    print(f"\nPredicted status: {status}")

if __name__ == "__main__":
    main()
