# Машинное обучение для мониторинга умного здания

Репозиторий содержит две лабораторные работы по дисциплине «Машинное обучение в системах искусственного интеллекта». Общая прикладная область — анализ данных умного здания и IoT-датчиков с применением методов машинного обучения.

## Состав проекта

```text
smart_building_ml/
├── lab1_smart_building_energy_prediction/
│   ├── data/
│   │   └── building_energy.csv
│   ├── models/
│   │   ├── best_energy_model.pkl
│   │   └── feature_columns.pkl
│   ├── reports/
│   │   ├── metrics.csv
│   │   ├── report.txt
│   │   ├── metrics_comparison.png
│   │   ├── actual_vs_predicted.png
│   │   └── residuals.png
│   ├── generate_data.py
│   ├── code.py
│   ├── predict.py
│   ├── requirements.txt
│   └── README.md
│
├── lab2_iot_sensor_anomaly_detection/
│   ├── data/
│   │   └── iot_sensor_states.csv
│   ├── models/
│   │   ├── scaler.pkl
│   │   ├── feature_columns.pkl
│   │   ├── kmeans_model.pkl
│   │   └── best_anomaly_model.pkl
│   ├── reports/
│   │   ├── silhouette_scores.csv
│   │   ├── clustered_sensor_states.csv
│   │   ├── anomaly_detection_metrics.csv
│   │   ├── state_classification_metrics.csv
│   │   ├── silhouette_scores.png
│   │   ├── pca_clusters.png
│   │   ├── pca_anomalies.png
│   │   ├── anomaly_confusion_matrix.png
│   │   └── report.txt
│   ├── generate_data.py
│   ├── project.py
│   ├── predict.py
│   ├── requirements.txt
│   └── README.md
│
├── Отчет_лабораторная_1_прогнозирование_энергопотребления.docx
├── Отчет_лабораторная_2_аномалии_IoT_датчиков.docx
└── Реферат_ML_умное_здание.docx
```

## Лабораторная работа 1

**Тема:** прогнозирование энергопотребления умного здания.

Цель работы — построить модель машинного обучения, которая по данным IoT-датчиков и эксплуатационным характеристикам здания прогнозирует часовое энергопотребление.

### Используемые признаки

- температура воздуха;
- влажность;
- уровень освещенности;
- количество людей в здании;
- активность HVAC-системы;
- активность освещения;
- час суток;
- день недели;
- рабочий/выходной день.

### Используемые модели

- Linear Regression;
- Random Forest Regressor;
- Gradient Boosting Regressor.

### Метрики качества

- MAE;
- RMSE;
- R2;
- MAPE.

После запуска формируются графики сравнения моделей, фактического и прогнозного энергопотребления, а также график остатков.

## Лабораторная работа 2

**Тема:** выявление аномалий IoT-датчиков умного здания.

Цель работы — выделить типовые режимы работы здания и обнаружить аномальные состояния датчиков, которые могут указывать на сбои оборудования, некорректные измерения или нештатное поведение системы.

### Используемые признаки

- температура;
- влажность;
- уровень CO2;
- энергопотребление;
- уровень освещенности;
- вибрация оборудования;
- уровень шума;
- активность HVAC-системы;
- количество людей в помещении.

### Используемые методы

- KMeans;
- Agglomerative Clustering;
- PCA;
- Isolation Forest;
- Local Outlier Factor;
- One-Class SVM;
- Random Forest Classifier.

### Метрики качества

- Silhouette Score;
- Accuracy;
- Precision;
- Recall;
- F1-score;
- ROC-AUC.

После запуска формируются результаты кластеризации, графики PCA, таблицы метрик и итоговый текстовый отчет.

## Установка зависимостей

Для каждой лабораторной зависимости указаны в отдельном файле `requirements.txt`.

Пример установки:

```bash
pip install -r requirements.txt
```

## Запуск лабораторной работы 1

```bash
cd lab1_smart_building_energy_prediction
pip install -r requirements.txt
python generate_data.py
python code.py
python predict.py
```

## Запуск лабораторной работы 2

```bash
cd lab2_iot_sensor_anomaly_detection
pip install -r requirements.txt
python generate_data.py
python project.py
python predict.py
```

## Результаты

В архивах уже присутствуют данные, обученные модели и результаты выполнения. Повторный запуск скриптов перегенерирует датасеты, модели, графики и отчеты.

## Используемые библиотеки

- pandas;
- numpy;
- scikit-learn;
- matplotlib;
- joblib.

## Назначение проекта

Проект демонстрирует применение методов машинного обучения для задач мониторинга умного здания: прогнозирования энергопотребления, анализа режимов работы IoT-инфраструктуры и выявления аномальных состояний датчиков.
