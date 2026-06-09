# Лабораторная работа 1. Прогнозирование энергопотребления умного здания

Проект посвящен задаче регрессии: по данным IoT-датчиков и характеристикам эксплуатации здания необходимо предсказать энергопотребление за час.

## Структура

```text
lab1_smart_building_energy_prediction/
├── data/
│   └── building_energy.csv
├── models/
├── reports/
├── generate_data.py
├── code.py
├── predict.py
├── requirements.txt
└── README.md
```

## Запуск

```bash
pip install -r requirements.txt
python generate_data.py
python code.py
python predict.py
```

## Используемые модели

- Linear Regression;
- Random Forest Regressor;
- Gradient Boosting Regressor.

## Метрики

- MAE;
- RMSE;
- R2;
- MAPE.

## Результаты

После запуска в папке `reports/` сохраняются таблица метрик, текстовый отчет и графики. В папке `models/` сохраняется лучшая обученная модель.
