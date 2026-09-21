# 🚢 Module 1 — Spaceship Titanic: EDA + бинарная классификация

**Автор:** Гурьевская Валерия Евгеньевна, группа ПКТб-23-1  
**Дисциплина:** Машинное обучение и ИИ  
**Дата сдачи:** 2026-09-22  
**Датасет:** [Spaceship Titanic (Kaggle Competition)](https://www.kaggle.com/c/spaceship-titanic)  

---

## 📊 Результаты моделей

| Модель | Accuracy | Precision | Recall | F1 | ROC-AUC | Время обучения |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7918 | 0.7862 | 0.8059 | 0.7959 | 0.8841 | 0.0263 сек |
| Decision Tree | 0.7999 | 0.7927 | 0.8162 | 0.8043 | 0.8725 | 0.0244 сек |
| **Random Forest (Бонус)** | **0.7993** | **0.8011** | **0.8002** | **0.8007** | **0.8915** | 0.8355 сек |

> [!NOTE]
> Все модели преодолели установленный целевой порог $ROC\text{-}AUC \ge 0.80$. Лучший результат по метрике площади под ROC-кривой показала ансамблевая модель **Random Forest** ($ROC\text{-}AUC = 0.8915$).

---

## 🚀 Быстрый старт (инференс из репозитория)

```python
import io
import joblib
import requests
import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/GuryevskayaValeriya/ml-course-guryevskaya/main/module-1-titanic"

# 1. Загрузка обученной модели Random Forest
model_url = f"{BASE_URL}/models/rf_model.pkl"
rf_model = joblib.load(io.BytesIO(requests.get(model_url).content))

# 2. Загрузка списка признаков
features_url = f"{BASE_URL}/models/feature_cols.json"
feature_cols = requests.get(features_url).json()

print(f"✅ Модель успешно загружена! Количество признаков: {len(feature_cols)}")
```

---

## 📁 Структура модуля

```text
module-1-titanic/
├── README.md               # Отчёт и документация по первому модулю
├── notebook.ipynb          # Полный Jupyter Notebook (все 14 обязательных разделов)
├── data/
│   ├── train.csv           # Обучающая выборка (8693 объекта)
│   ├── test.csv            # Тестовая выборка (4277 объектов)
│   └── titanic_info.md     # Подробное описание признаков и пропусков
├── models/
│   ├── lr_model.pkl        # Веса Logistic Regression
│   ├── dt_model.pkl        # Веса Decision Tree
│   ├── rf_model.pkl        # Веса Random Forest (бонусная ансамблевая модель)
│   ├── scaler.pkl          # Обученный StandardScaler
│   ├── feature_cols.json   # Точный порядок и список входных признаков
│   └── metrics.json        # Итоговые метрики моделей и время обучения
├── examples/
│   ├── eda_plots.png       # Аналитическая сетка графиков первичного анализа
│   ├── confusion_matrices.png # Матрицы ошибок классификаторов
│   └── roc_curves.png      # График сравнительных ROC-кривых
└── requirements.txt        # Список зависимостей с зафиксированными версиями
```
