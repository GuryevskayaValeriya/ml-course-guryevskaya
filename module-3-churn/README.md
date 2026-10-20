# 📉 Module 3 — Customer Churn: Ансамблевые методы и интерпретация

**Автор:** Гурьевская Валерия Евгеньевна, группа ПКТб-23-1  
**Дисциплина:** Машинное обучение и ИИ  
**Дата сдачи:** 2026-10-20  
**Датасет:** [Kaggle: Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM Sample Data Sets)  

---

## 📊 Результаты моделей на тестовой выборке

| Модель | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Время обучения |
|---|---|---|---|---|---|---|
| Random Forest | 0.7615 | 0.5365 | 0.7460 | 0.6242 | 0.8388 | 0.387 сек |
| XGBoost | 0.7665 | 0.5468 | 0.7032 | 0.6152 | 0.8301 | 0.317 сек |
| LightGBM (Baseline) | 0.7601 | 0.5345 | 0.7460 | 0.6228 | 0.8384 | 0.127 сек |
| CatBoost | 0.7488 | 0.5187 | 0.7433 | 0.6110 | 0.8409 | 0.723 сек |
| **LightGBM (Optuna Tuned)** | **0.7438** | **0.5111** | **0.8021** | **0.6243** | **0.8465** | **0.136 сек** |
| **Stacking Classifier (Бонус)** | **0.8006** | **0.6380** | **0.5749** | **0.6048** | **0.8467** | **2.111 сек** |

> [!NOTE]
> Все обученные ансамблевые модели преодолели порог $\text{ROC-AUC} \ge 0.830$.  
> Лучший баланс скорости и качества показала модель **LightGBM**, оптимизированная с помощью **Optuna** ($\text{ROC-AUC} = 0.8465$, полнота $\text{Recall} = 0.8021$ — обнаружено более $80\%$ уходящих абонентов).  
> В качестве продвинутого бонуса реализован ансамбль **Stacking Classifier** (Random Forest + XGBoost + CatBoost с мета-классификатором Logistic Regression), достигший наивысшей точности $\text{Accuracy} = 0.8006$ и точности предсказания $\text{Precision} = 0.6380$.

---

## 🚀 Быстрый старт (инференс из GitHub репозитория)

```python
import io
import json
import joblib
import requests
import numpy as np
import pandas as pd

# Базовый URL сырых файлов репозитория
BASE_URL = "https://raw.githubusercontent.com/GuryevskayaValeriya/ml-course-guryevskaya/main/module-3-churn"

# 1. Загрузка лучшей модели (LightGBM Optuna)
model_url = f"{BASE_URL}/models/best_model.pkl"
best_model = joblib.load(io.BytesIO(requests.get(model_url).content))

# 2. Загрузка scaler
scaler_url = f"{BASE_URL}/models/scaler.pkl"
scaler = joblib.load(io.BytesIO(requests.get(scaler_url).content))

# 3. Загрузка списка признаков
features_url = f"{BASE_URL}/models/feature_cols.json"
feature_cols = requests.get(features_url).json()

# 4. Пример профиля абонента для скоринга риска оттока
client = {
    'gender': 1, 'SeniorCitizen': 0, 'Partner': 0, 'Dependents': 0,
    'tenure': 2, 'PhoneService': 1, 'MultipleLines': 0, 'InternetService': 1,
    'OnlineSecurity': 0, 'OnlineBackup': 0, 'DeviceProtection': 0,
    'TechSupport': 0, 'StreamingTV': 1, 'StreamingMovies': 1,
    'Contract': 0, 'PaperlessBilling': 1, 'PaymentMethod': 2,
    'MonthlyCharges': 85.5, 'TotalCharges': 171.0, 'TenureYears': 0.1667,
    'AvgMonthlyCharge': 57.0, 'IsNewClient': 1, 'HasMultipleServices': 4
}

df_client = pd.DataFrame([client])[feature_cols]
client_scaled = scaler.transform(df_client)

prob_churn = best_model.predict_proba(client_scaled)[0, 1]
is_churn = int(prob_churn >= 0.5)

print(f"Вероятность оттока абонента: {prob_churn:.2%}")
print(f"Вердикт системы: {'Клиент в зоне высокого риска (Уйдет)' if is_churn else 'Клиент стабилен (Останется)'}")
```

---

## 📁 Структура модуля

```
module-3-churn/
├── README.md                      # Полное описание модуля, результатов и инференса
├── notebook.ipynb                 # Полный интерактивный отчет (14 разделов, выполненные ячейки)
├── requirements.txt               # Список зависимостей для воспроизводимости
├── data/
│   ├── telco_churn.csv            # Исходный датасет Telco Churn (7043 строки, 21 колонка)
│   └── churn_info.md              # Подробная спецификация признаков и распределений
├── models/
│   ├── best_model.pkl             # Оптимизированная модель LightGBM (Optuna)
│   ├── rf_model.pkl               # Random Forest Classifier (Bagging)
│   ├── xgb_model.pkl              # XGBoost Classifier
│   ├── lgbm_model.pkl             # Базовый LightGBM Classifier
│   ├── cat_model.pkl              # CatBoost Classifier
│   ├── stacking_model.pkl         # Бонусный Stacking Classifier (RF+XGB+Cat)
│   ├── scaler.pkl                 # StandardScaler числовых признаков
│   ├── label_encoders.pkl         # Словарь кодировщиков категориальных признаков
│   ├── feature_cols.json          # 23 признака (с учетом Feature Engineering)
│   └── metrics.json               # Структурированные метрики и лучшие гиперпараметры Optuna
└── examples/
    ├── eda_plots.png              # Комплексный разведочный анализ данных (сетка 2x3)
    ├── roc_curves.png             # Совместные ROC-кривые всех 4 ансамблей
    ├── feature_importance.png     # Топ-15 важнейших признаков модели
    ├── shap_summary.png           # SHAP Summary Plot (глобальное влияние факторов)
    └── shap_waterfall.png         # SHAP Waterfall Plot (персональный аудит клиента)
```

---

## 💡 Ключевые бизнес-инсайты (SHAP & Feature Importance)
1. **Тип контракта (`Contract`):** Помесячный контракт — главный драйвер оттока (вероятность ухода достигает $42.7\%$, тогда как двухлетний контракт удерживает $97.2\%$ клиентов).
2. **Срок жизни клиента (`tenure`):** Первые 6 месяцев обслуживания являются критическим интервалом наибольшей уязвимости базы абонентов.
3. **Финансовая нагрузка (`MonthlyCharges`, `AvgMonthlyCharge`):** Дорогие пакетные тарифы оптоволоконного интернета (`Fiber optic`) вызывают повышенный отток при отсутствии сопутствующих сервисов заботы.
4. **Сервисы удержания (`OnlineSecurity`, `TechSupport`):** Наличие выделенной техподдержки и кибербезопасности снижает вероятность ухода абонента практически вдвое.
