# Лабораторная работа 2. Выявление аномалий IoT-датчиков умного здания

Проект посвящен анализу данных датчиков умного здания. Цель - выделить типовые режимы работы здания и автоматически обнаруживать аномальные состояния датчиков.

## Структура

```text
lab2_iot_sensor_anomaly_detection/
├── data/
│   └── iot_sensor_states.csv
├── models/
├── reports/
├── generate_data.py
├── project.py
├── predict.py
├── requirements.txt
└── README.md
```

## Запуск

```bash
pip install -r requirements.txt
python generate_data.py
python project.py
python predict.py
```

## Используемые методы

- KMeans;
- Agglomerative Clustering;
- PCA;
- Isolation Forest;
- Local Outlier Factor;
- One-Class SVM;
- Random Forest Classifier.

## Метрики

- Silhouette Score;
- Accuracy;
- Precision;
- Recall;
- F1-score;
- ROC-AUC.

## Результаты

После запуска в папке `reports/` сохраняются графики, таблицы метрик и итоговый отчет. В папке `models/` сохраняются обученные модели и scaler.
