from pathlib import Path
import subprocess
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/building_energy.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

FEATURE_COLUMNS = [
    "hour", "day_of_week", "is_weekend", "outside_temperature", "humidity",
    "occupancy_count", "lighting_usage", "equipment_load_kw", "hvac_load_kw",
    "co2_ppm", "active_rooms"
]
TARGET_COLUMN = "energy_consumption_kwh"


def ensure_data():
    if not DATA_PATH.exists():
        subprocess.run([sys.executable, "generate_data.py"], check=True)


def mape(y_true, y_pred):
    y_true = pd.Series(y_true).replace(0, 1e-9)
    return (abs((y_true - y_pred) / y_true).mean()) * 100


def main():
    ensure_data()
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=250, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    results = []
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        results.append({
            "model": name,
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": rmse,
            "R2": r2_score(y_test, y_pred),
            "MAPE": mape(y_test.reset_index(drop=True), pd.Series(y_pred)),
        })

    results_df = pd.DataFrame(results).sort_values("RMSE")
    results_df.to_csv(REPORTS_DIR / "metrics.csv", index=False)

    best_name = results_df.iloc[0]["model"]
    best_model = models[best_name]
    best_pred = predictions[best_name]

    joblib.dump(best_model, MODELS_DIR / "best_energy_model.pkl")
    joblib.dump(FEATURE_COLUMNS, MODELS_DIR / "feature_columns.pkl")

    with open(REPORTS_DIR / "report.txt", "w", encoding="utf-8") as f:
        f.write("Energy consumption prediction report\n")
        f.write("====================================\n\n")
        f.write(f"Dataset size: {df.shape}\n")
        f.write(f"Best model: {best_name}\n\n")
        f.write(results_df.to_string(index=False))

    plt.figure(figsize=(8, 5))
    sns.barplot(data=results_df, x="model", y="RMSE", color="steelblue")
    plt.title("Model comparison by RMSE")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "metrics_comparison.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, best_pred, alpha=0.55)
    min_v, max_v = min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")
    plt.xlabel("Actual energy consumption, kWh")
    plt.ylabel("Predicted energy consumption, kWh")
    plt.title(f"Actual vs predicted: {best_name}")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "actual_vs_predicted.png", dpi=200)
    plt.close()

    residuals = y_test.reset_index(drop=True) - pd.Series(best_pred)
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=30, kde=True, color="steelblue")
    plt.title("Prediction residuals")
    plt.xlabel("Residual, kWh")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "residuals.png", dpi=200)
    plt.close()

    if hasattr(best_model, "feature_importances_"):
        importance = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": best_model.feature_importances_})
        importance = importance.sort_values("importance", ascending=False)
        importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
        plt.figure(figsize=(8, 5))
        sns.barplot(data=importance, x="importance", y="feature", color="steelblue")
        plt.title("Feature importance")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "feature_importance.png", dpi=200)
        plt.close()

    print("Best model:", best_name)
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()
