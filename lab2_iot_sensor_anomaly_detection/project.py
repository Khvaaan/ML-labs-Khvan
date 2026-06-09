from pathlib import Path
import subprocess
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

DATA_PATH = Path("data/iot_sensor_states.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

FEATURE_COLUMNS = [
    "temperature", "humidity", "co2_ppm", "motion_level", "power_kw",
    "occupancy_count", "vibration_level", "network_latency_ms",
    "packet_loss_percent", "battery_level"
]


def ensure_data():
    if not DATA_PATH.exists():
        subprocess.run([sys.executable, "generate_data.py"], check=True)


def convert_labels(pred):
    # sklearn anomaly models usually return 1 for normal and -1 for anomaly
    return pd.Series(pred).map({1: 0, -1: 1}).values


def main():
    ensure_data()
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df["is_anomaly"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(FEATURE_COLUMNS, MODELS_DIR / "feature_columns.pkl")

    silhouette_rows = []
    for k in range(2, 8):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_rows.append({"k": k, "silhouette_score": score})
    sil_df = pd.DataFrame(silhouette_rows)
    sil_df.to_csv(REPORTS_DIR / "silhouette_scores.csv", index=False)
    optimal_k = int(sil_df.sort_values("silhouette_score", ascending=False).iloc[0]["k"])

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    agg_labels = AgglomerativeClustering(n_clusters=optimal_k).fit_predict(X_scaled)
    df["cluster"] = cluster_labels
    df["agg_cluster"] = agg_labels
    df.to_csv(REPORTS_DIR / "clustered_sensor_states.csv", index=False)
    joblib.dump(kmeans, MODELS_DIR / "kmeans_model.pkl")

    plt.figure(figsize=(7, 5))
    sns.lineplot(data=sil_df, x="k", y="silhouette_score", marker="o", color="steelblue")
    plt.title("Silhouette Score for KMeans")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "silhouette_scores.png", dpi=200)
    plt.close()

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "cluster": cluster_labels, "is_anomaly": y})

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="cluster", palette="tab10", s=30)
    plt.title("KMeans clusters in PCA space")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "pca_clusters.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="is_anomaly", s=30, palette="tab10")
    plt.title("True anomalies in PCA space")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "pca_anomalies.png", dpi=200)
    plt.close()

    contamination = y.mean()
    anomaly_models = {
        "Isolation Forest": IsolationForest(contamination=contamination, random_state=42),
        "Local Outlier Factor": LocalOutlierFactor(contamination=contamination),
        "One-Class SVM": OneClassSVM(nu=contamination, kernel="rbf", gamma="scale"),
    }

    anomaly_results = []
    anomaly_predictions = {}
    for name, model in anomaly_models.items():
        if name == "Local Outlier Factor":
            raw_pred = model.fit_predict(X_scaled)
            scores = -model.negative_outlier_factor_
        else:
            model.fit(X_scaled)
            raw_pred = model.predict(X_scaled)
            scores = -model.decision_function(X_scaled)
        pred = convert_labels(raw_pred)
        anomaly_predictions[name] = pred
        anomaly_results.append({
            "model": name,
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1_score": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, scores),
        })

    anomaly_df = pd.DataFrame(anomaly_results).sort_values("f1_score", ascending=False)
    anomaly_df.to_csv(REPORTS_DIR / "anomaly_detection_metrics.csv", index=False)
    best_anomaly_name = anomaly_df.iloc[0]["model"]

    if best_anomaly_name != "Local Outlier Factor":
        joblib.dump(anomaly_models[best_anomaly_name], MODELS_DIR / "best_anomaly_model.pkl")
    else:
        # для predict.py используем Isolation Forest, так как LOF в стандартном режиме не предназначен для новых объектов
        iso = IsolationForest(contamination=contamination, random_state=42)
        iso.fit(X_scaled)
        joblib.dump(iso, MODELS_DIR / "best_anomaly_model.pkl")
        best_anomaly_name = "Isolation Forest"

    cm = confusion_matrix(y, anomaly_predictions[anomaly_df.iloc[0]["model"]])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["normal", "anomaly"], yticklabels=["normal", "anomaly"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Anomaly detection confusion matrix")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "anomaly_confusion_matrix.png", dpi=200)
    plt.close()

    # Дополнительно: классификация состояния здания по нормальным записям
    normal_df = df[df["is_anomaly"] == 0].copy()
    X_norm = normal_df[FEATURE_COLUMNS]
    y_norm = normal_df["building_state"]
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y_norm, test_size=0.25, random_state=42, stratify=y_norm)
    state_scaler = StandardScaler()
    X_train_scaled = state_scaler.fit_transform(X_train)
    X_test_scaled = state_scaler.transform(X_test)
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }
    clf_rows = []
    for name, clf in classifiers.items():
        clf.fit(X_train_scaled, y_train)
        pred = clf.predict(X_test_scaled)
        clf_rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "f1_score": f1_score(y_test, pred, average="weighted"),
        })
    clf_df = pd.DataFrame(clf_rows).sort_values("f1_score", ascending=False)
    clf_df.to_csv(REPORTS_DIR / "state_classification_metrics.csv", index=False)

    with open(REPORTS_DIR / "report.txt", "w", encoding="utf-8") as f:
        f.write("IoT sensor anomaly detection report\n")
        f.write("===================================\n\n")
        f.write(f"Dataset size: {df.shape}\n")
        f.write(f"Anomaly share: {contamination:.3f}\n")
        f.write(f"Optimal number of KMeans clusters: {optimal_k}\n")
        f.write(f"Best anomaly model for saved prediction: {best_anomaly_name}\n\n")
        f.write("Silhouette scores:\n")
        f.write(sil_df.to_string(index=False))
        f.write("\n\nAnomaly detection metrics:\n")
        f.write(anomaly_df.to_string(index=False))
        f.write("\n\nState classification metrics:\n")
        f.write(clf_df.to_string(index=False))

    print("Optimal k:", optimal_k)
    print(anomaly_df.to_string(index=False))
    print(clf_df.to_string(index=False))

if __name__ == "__main__":
    main()
