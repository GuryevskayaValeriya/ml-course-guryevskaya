## Описание данных

**Источник:** Kaggle Competition — Spaceship Titanic (https://www.kaggle.com/c/spaceship-titanic)  

**Структура датасета:**
- Всего примеров в обучающей выборке (`train.csv`): 8693
- Всего примеров в тестовой выборке (`test.csv`): 4277
- Признаков: 13 независимых признаков + целевая переменная `Transported`
- Числовые признаки (6): `Age`, `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck` 
- Категориальные / текстовые признаки (7) : `PassengerId`, `HomePlanet`, `CryoSleep`, `Cabin`, `Destination`, `VIP`, `Name`
- Целевой признак: `Transported` (`True` / `False`)

### Описание колонок Spaceship Titanic:

| Столбец | Тип | Описание | Доля пропусков |
|---|---|---|---|
| `PassengerId` | object | Уникальный идентификатор пассажира формата `gggg_pp` | 0.0% |
| `HomePlanet` | object | Планета отправления (Earth, Europa, Mars) | ~2.3% |
| `CryoSleep` | object | Находился ли пассажир в криосне (True/False) | ~2.5% |
| `Cabin` | object | Каюта формата `Deck/Num/Side` (борт P - Port или S - Starboard) | ~2.3% |
| `Destination` | object | Планета назначения | ~2.1% |
| `Age` | float64 | Возраст пассажира | ~2.1% |
| `VIP` | object | Оплачен ли специальный VIP-сервис | ~2.4% |
| `RoomService` | float64 | Сумма расходов пассажира на обслуживание в каюте | ~2.1% |
| `FoodCourt` | float64 | Расходы на фудкорт | ~2.1% |
| `ShoppingMall` | float64 | Расходы на торговый центр | ~2.4% |
| `Spa` | float64 | Расходы на спа-салон | ~2.1% |
| `VRDeck` | float64 | Расходы на шлем виртуальной реальности | ~2.2% |
| `Name` | object | Имя и фамилия пассажира | ~2.3% |
| `Transported` | bool | Целевая метка: был ли пассажир перемещен в другое измерение | 0.0% |

- Размер train_df: (8693, 14)
- Размер test_df:  (4277, 13)


Распределение целевой переменной Transported:
- Класс True: 4378 объектов (50.36%)
- Класс False: 4315 объектов (49.64%)

**Файл данных в репозитории:** `module-1-titanic/data/titanic_info.md`