# 📉 Датасет «Telco Customer Churn»

## 📌 Общие сведения
- **Источник:** [Kaggle: Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM Sample Data Sets)
- **Тип задачи:** Supervised Learning — Обучение с учителем, бинарная классификация (Binary Classification)
- **Целевая переменная:** `Churn` — факт ухода клиента из телекоммуникационной компании в течение последнего месяца (`Yes` / `No`).

---

## 📊 Структура и размерность данных
- **Всего наблюдений:** 7 043 строки
- **Всего столбцов:** 21 (включая идентификатор `customerID` и целевую переменную `Churn`)
- **Предикторов:** 19 признаков
  - **Числовые признаки (Numerical):** 3 признака (`tenure`, `MonthlyCharges`, `TotalCharges`)
  - **Категориальные / бинарные признаки (Categorical/Binary):** 16 признаков (`gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`)

---

## 🎯 Распределение целевой переменной (`Churn`)
- **No (клиент остался):** 5 174 (73.46%)
- **Yes (клиент ушёл):** 1 869 (26.54%)
- **Соотношение классов:** примерно 2.77 : 1 (умеренный дисбаланс классов).
- **Вывод для моделирования:** Для корректного обучения алгоритмов классификации необходимо учитывать дисбаланс: использовать стратифицированное разбиение (`stratify=y`), балансировку весов классов (`class_weight='balanced'`, `scale_pos_weight`), а также ориентироваться на метрики `ROC-AUC`, `Precision`, `Recall` и `F1-score` вместо `Accuracy`.

---

## 🔍 Группы признаков и их смысловое описание

### 1. Демографические признаки (Demographics)
- `customerID`: Уникальный буквенно-числовой идентификатор клиента (удаляется при обучении моделей).
- `gender`: Пол клиента (`Male`, `Female`).
- `SeniorCitizen`: Является ли клиент пенсионером (`1` — да, `0` — нет).
- `Partner`: Наличие супруга / партнёра (`Yes`, `No`).
- `Dependents`: Наличие иждивенцев / детей (`Yes`, `No`).

### 2. Подключённые телекоммуникационные сервисы (Services)
- `PhoneService`: Подключение домашнего телефона (`Yes`, `No`).
- `MultipleLines`: Наличие нескольких телефонных линий (`Yes`, `No`, `No phone service`).
- `InternetService`: Тип интернет-провайдера (`DSL`, `Fiber optic`, `No`).
- `OnlineSecurity`: Подключение услуги онлайн-безопасности (`Yes`, `No`, `No internet service`).
- `OnlineBackup`: Подключение услуги резервного облачного копирования (`Yes`, `No`, `No internet service`).
- `DeviceProtection`: Страхование / защита оборудования (`Yes`, `No`, `No internet service`).
- `TechSupport`: Выделенная техническая поддержка (`Yes`, `No`, `No internet service`).
- `StreamingTV`: Стриминговое телевидение (`Yes`, `No`, `No internet service`).
- `StreamingMovies`: Стриминговый кинотеатр (`Yes`, `No`, `No internet service`).

### 3. Параметры контракта и биллинга (Account & Billing)
- `tenure`: Количество месяцев, в течение которых клиент пользуется услугами компании (от 0 до 72 месяцев).
- `Contract`: Условия контракта (`Month-to-month` — помесячный, `One year` — годовой, `Two year` — двухлетний).
- `PaperlessBilling`: Электронный безналичный счёт (`Yes`, `No`).
- `PaymentMethod`: Способ оплаты (`Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)`).
- `MonthlyCharges`: Текущая ежемесячная сумма платежа в долларах США ($18.25 – $118.75).
- `TotalCharges`: Совокупная сумма, выплаченная клиентом за всё время обслуживания. Хранится в исходном виде как строка из-за наличия 11 пробельных значений у новых клиентов (`tenure = 0`), требует преобразования в вещественный тип (`float`) и заполнения пропусков.
