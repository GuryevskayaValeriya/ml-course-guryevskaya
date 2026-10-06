# 🏠 Module 2 — House Prices: Предобработка данных и регрессия

**Автор:** Гурьевская Валерия Евгеньевна, группа ПКТб-23-1  
**Дисциплина:** Машинное обучение и ИИ  
**Дата сдачи:** 2026-10-06  
**Датасет:** [Kaggle: House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)  

---

## 📊 Результаты моделей на тестовой выборке

| Модель | RMSE (log) | MAE (log) | $R^2$ | RMSE ($) | MAE ($) | Время обучения |
|---|---|---|---|---|---|---|
| Linear Regression | 0.1321 | 0.0922 | 0.8964 | $22,479 | $15,681 | 0.031 сек |
| Ridge ($lpha=50.0$) | 0.1292 | 0.0890 | 0.9010 | $22,117 | $15,294 | 0.173 сек |
| Lasso ($lpha=0.001$) | 0.1254 | 0.0854 | 0.9066 | $20,907 | $14,573 | 2.372 сек |
| ElasticNet ($lpha=0.002$) | 0.1242 | 0.0845 | 0.9085 | $20,571 | $14,390 | 3.374 сек |
| Random Forest (Бонус) | 0.1380 | 0.0918 | 0.8870 | $23,683 | $15,839 | 0.573 сек |
| XGBoost (Бонус) | 0.1218 | 0.0824 | 0.9120 | $20,359 | $14,442 | 0.555 сек |
| **Stacking Regressor (Лучшая)** | **0.1197** | **0.0804** | **0.9151** | **$19,467** | **$13,732** | **5.573 сек** |

> [!NOTE]
> Все обученные регуляризованные и ансамблевые модели преодолели порог $R^2 \ge 0.90$.  
> Лучший результат показал ансамбль **Stacking Regressor** ($R^2 = 0.9151$, RMSE в долларах = $19,467$).  
> Среди линейных моделей лучшей оказалась **Lasso** ($R^2 = 0.9066$), которая занулила 60 неинформативных признаков из 220.

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
BASE_URL = "https://raw.githubusercontent.com/GuryevskayaValeriya/ml-course-guryevskaya/main/module-2-house-prices"

# 1. Загрузка лучшей модели (Lasso/Stacking)
model_url = f"{BASE_URL}/models/lasso_model.pkl"
lasso_model = joblib.load(io.BytesIO(requests.get(model_url).content))

# 2. Загрузка scaler
scaler_url = f"{BASE_URL}/models/scaler.pkl"
scaler = joblib.load(io.BytesIO(requests.get(scaler_url).content))

# 3. Загрузка списка признаков
features_url = f"{BASE_URL}/models/feature_cols.json"
feature_cols = requests.get(features_url).json()

# 4. Предсказание цены нового дома
house = {
    'OverallQual': 9, 'TotalSF': 4600, 'GrLivArea': 2800,
    'YearBuilt': 2015, 'YearRemodAdd': 2015, 'YrSold': 2019,
    'TotalBath': 4.5, 'GarageCars': 3, 'Fireplaces': 2,
    'OverallQual_TotalSF': 41400, 'ExterQual': 4, 'KitchenQual': 5
}

df_new = pd.DataFrame([house])
for c in feature_cols:
    if c not in df_new.columns:
        df_new[c] = 0.0
df_new = df_new[feature_cols]

pred_log = lasso_model.predict(scaler.transform(df_new))[0]
pred_price = np.expm1(pred_log)
print(f"Оценочная стоимость дома: ${pred_price:,.0f}")
```

---

## 📁 Структура модуля

```
module-2-house-prices/
├── README.md                      # Описание модуля и метрик
├── notebook.ipynb                 # Полный отчет (14 разделов, выполненные ячейки)
├── requirements.txt               # Зависимости для воспроизведения
├── data/
│   ├── train.csv                  # Обучающая выборка (1460 строк)
│   ├── test.csv                   # Тестовая выборка (1459 строк)
│   ├── house_prices_info.md       # Подробная спецификация признаков
│   └── submission.csv             # Kaggle submission
├── models/
│   ├── best_model.pkl             # Лучшая модель (Stacking)
│   ├── lasso_model.pkl            # Lasso (L1)
│   ├── ridge_model.pkl            # Ridge (L2)
│   ├── enet_model.pkl             # ElasticNet
│   ├── lr_model.pkl               # Базовая Linear Regression
│   ├── xgb_model.pkl              # XGBoost Regressor
│   ├── stacking_model.pkl         # Stacking Regressor
│   ├── scaler.pkl                 # StandardScaler
│   ├── feature_cols.json          # 220 отобранных признаков
│   └── metrics.json               # Метрики всех моделей
└── examples/
    ├── eda_plots.png              # Графики разведочного анализа
    ├── residuals.png              # График и распределение остатков
    ├── predictions_scatter.png    # Реальные vs Предсказанные цены
    └── feature_importances.png    # Топ коэффициентов признаков
```
