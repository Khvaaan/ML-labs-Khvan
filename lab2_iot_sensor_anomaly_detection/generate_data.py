from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def normal_row(rng, state):
    if state == "night_low_load":
        occupancy = max(0, rng.normal(8, 4))
        power = rng.normal(18, 4)
        co2 = rng.normal(460, 35)
        motion = rng.normal(0.08, 0.04)
    elif state == "working_load":
        occupancy = max(10, rng.normal(95, 22))
        power = rng.normal(58, 9)
        co2 = rng.normal(850, 110)
        motion = rng.normal(0.66, 0.12)
    elif state == "evening_medium_load":
        occupancy = max(5, rng.normal(42, 12))
        power = rng.normal(36, 6)
        co2 = rng.normal(620, 70)
        motion = rng.normal(0.34, 0.09)
    else:
        occupancy = max(30, rng.normal(150, 25))
        power = rng.normal(82, 12)
        co2 = rng.normal(1120, 130)
        motion = rng.normal(0.88, 0.08)

    temperature = rng.normal(22.2, 1.4)
    humidity = rng.normal(46, 7)
    vibration = abs(rng.normal(0.12, 0.04))
    network_latency_ms = abs(rng.normal(32, 9))
    packet_loss = abs(rng.normal(0.25, 0.18))
    battery_level = rng.normal(78, 13)

    return {
        "temperature": round(temperature, 2),
        "humidity": round(np.clip(humidity, 20, 80), 2),
        "co2_ppm": round(np.clip(co2, 380, 1600), 1),
        "motion_level": round(np.clip(motion, 0, 1), 3),
        "power_kw": round(max(4, power), 2),
        "occupancy_count": int(max(0, occupancy)),
        "vibration_level": round(vibration, 3),
        "network_latency_ms": round(network_latency_ms, 2),
        "packet_loss_percent": round(np.clip(packet_loss, 0, 5), 3),
        "battery_level": round(np.clip(battery_level, 5, 100), 2),
        "building_state": state,
        "is_anomaly": 0,
    }


def anomaly_row(rng):
    kind = rng.choice(["co2_peak", "power_surge", "sensor_failure", "network_problem", "battery_critical"])
    base = normal_row(rng, rng.choice(["night_low_load", "working_load", "evening_medium_load", "event_high_load"]))

    if kind == "co2_peak":
        base["co2_ppm"] = round(rng.normal(2100, 250), 1)
        base["temperature"] = round(rng.normal(24, 2), 2)
    elif kind == "power_surge":
        base["power_kw"] = round(rng.normal(145, 18), 2)
        base["vibration_level"] = round(abs(rng.normal(0.55, 0.12)), 3)
    elif kind == "sensor_failure":
        base["temperature"] = round(rng.choice([rng.normal(-5, 3), rng.normal(48, 4)]), 2)
        base["humidity"] = round(rng.choice([rng.normal(7, 3), rng.normal(96, 3)]), 2)
    elif kind == "network_problem":
        base["network_latency_ms"] = round(rng.normal(260, 45), 2)
        base["packet_loss_percent"] = round(rng.normal(12, 3), 3)
    elif kind == "battery_critical":
        base["battery_level"] = round(rng.normal(4, 1.8), 2)
        base["network_latency_ms"] = round(rng.normal(120, 35), 2)

    base["building_state"] = "anomaly_" + kind
    base["is_anomaly"] = 1
    return base


def main():
    rng = np.random.default_rng(123)
    rows = []
    states = ["night_low_load", "working_load", "evening_medium_load", "event_high_load"]
    probs = [0.27, 0.43, 0.22, 0.08]

    for _ in range(750):
        state = rng.choice(states, p=probs)
        rows.append(normal_row(rng, state))
    for _ in range(70):
        rows.append(anomaly_row(rng))

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(DATA_DIR / "iot_sensor_states.csv", index=False)
    print("Saved data/iot_sensor_states.csv", df.shape)

if __name__ == "__main__":
    main()
