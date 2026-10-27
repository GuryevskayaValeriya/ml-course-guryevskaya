# 🛒 Module 4 — Online Shoppers: Инженерия признаков и ансамблевые модели

**Автор:** Гурьевская Валерия Евгеньевна, группа ПКТб-23-1  
**Дисциплина:** Машинное обучение и ИИ  
**Дата сдачи:** 2026-10-27  
**Датасет:** [UCI / Kaggle: Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)

---

## 📊 Результаты моделей на тестовой выборке

| Модель | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Время обучения |
|---|---|---|---|---|---|---|
| Random Forest | 0.8662 | 0.5469 | 0.7932 | 0.6474 | 0.9295 | 0.370 сек |
| XGBoost | 0.8796 | 0.5906 | 0.7251 | 0.6510 | 0.9224 | 0.370 сек |
| LightGBM (Baseline) | 0.8759 | 0.5725 | 0.7853 | 0.6623 | 0.9277 | 0.148 сек |
| CatBoost | 0.8755 | 0.5698 | 0.8010 | 0.6659 | 0.9269 | 0.872 сек |
| **LightGBM (Optuna Tuned)** | **0.8617** | **0.5378** | **0.8246** | **0.6510** | **0.9323** | **0.170 сек** |

> [!NOTE]
> Все построенные ансамблевые модели преодолели установленный порог $\text{ROC-AUC} \ge 0.920$.  
> Лучший баланс скорости обучения и обобщающей способности показала модель **LightGBM**, настроенная с помощью **Optuna** (на кросс-валидации $\text{CV ROC-AUC} = 0.9352$, на тестовой выборке $\text{ROC-AUC} = 0.9323$, полнота обнаружения покупателей $\text{Recall} = 0.8246$).

---

## 🚀 Быстрый старт (инференс из GitHub репозитория)

```python
import io
import json
import joblib
import requests
import numpy as np
import pandas as pd

# базовый url репозитория
BASE_URL = "https://raw.githubusercontent.com/GuryevskayaValeriya/ml-course-guryevskaya/main/module-4-features"

# 1. загрузка лучшей модели (lightgbm optuna tuned)
model_url = f"{BASE_URL}/models/best_model.pkl"
best_model = joblib.load(io.BytesIO(requests.get(model_url).content))

# 2. загрузка scaler
scaler_url = f"{BASE_URL}/models/scaler.pkl"
scaler = joblib.load(io.BytesIO(requests.get(scaler_url).content))

# 3. загрузка списка признаков
features_url = f"{BASE_URL}/models/feature_cols.json"
feature_cols = requests.get(features_url).json()

# 4. пример профиля сессии пользователя
session = pd.DataFrame([{col: 0.0 for col in feature_cols}])
session['PageValues'] = 18.5
session['TotalDuration'] = 1420.0
session['ProductRelated'] = 32
session['TotalPages'] = 38
session['AvgPageValue'] = session['PageValues'] / (session['TotalPages'] + 1)
session['MonthNum'] = 11

# 5. прогноз вероятности покупки
proba = best_model.predict_proba(session)[:, 1][0]
print(f"Вероятность совершения покупки: {proba:.4f}")
```

---

## 📁 Структура модуля

```
module-4-features/
├── README.md                      ← Текущий отчёт по модулю
├── notebook.ipynb                 ← Полный интерактивный Jupyter-ноутбук (14 разделов)
├── requirements.txt               ← Зависимости Python
├── data/
│   ├── online_shoppers_intention.csv ← Исходный датасет (12 330 записей)
│   └── dataset_info.md            ← Паспорт датасета
├── models/
│   ├── best_model.pkl             ← Лучшая модель (LightGBM Optuna Tuned)
│   ├── rf_model.pkl               ← Random Forest
│   ├── xgb_model.pkl              ← XGBoost
│   ├── lgbm_model.pkl             ← LightGBM Baseline
│   ├── cat_model.pkl              ← CatBoost
│   ├── scaler.pkl                 ← StandardScaler
│   ├── feature_cols.json          ← Список входных признаков (34 фичи)
│   └── metrics.json               ← Сводные метрики всех моделей
└── examples/
    ├── eda_plots.png              ← Сетка из 6 графиков исследовательского анализа
    ├── roc_curves.png             ← Сравнение ROC-кривых 4 моделей
    ├── feature_importance.png     ← Важность признаков (LightGBM vs Random Forest)
    ├── shap_summary.png           ← Глобальный SHAP Summary plot
    ├── shap_waterfall.png         ← Локальный SHAP Waterfall plot для клиента
    └── shap_dependence.png        ← Нелинейная зависимость SHAP по признаку PageValues
```

---

## 💡 Ключевые выводы и инсайты

1. **Инженерия признаков (Feature Engineering):**  
   Создание агрегированных признаков (`TotalPages`, `TotalDuration`, `AvgPageValue`, `PageValue_TotalDuration`) позволило объединить разрозненные действия пользователя в интегральные метрики заинтересованности. Признак `AvgPageValue` вошел в топ-5 важнейших предикторов.
2. **Влияние метрик Google Analytics:**  
   Показатель ценности страницы `PageValues` выступает главным катализатором вероятности покупки (SHAP-значения резко возрастают при `PageValues > 0`). В то же время высокие показатели `ExitRates` и `BounceRates` свидетельствуют о раннем уходе пользователя и значительно снижают вероятность конверсии.
3. **Сезонность:**  
   Ноябрь (`Month_Nov`) характеризуется максимальной долей конверсий (~25%), что отражает влияние глобальных распродаж («Черная пятница», «Киберпонедельник»).
4. **Бизнес-применение:**  
   Модель позволяет в режиме реального времени выделять сессии с вероятностью покупки от 40% до 70% и направлять на них точечные механики конвертации (всплывающий промокод, персональное предложение, помощь оператора).
