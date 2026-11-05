# Don'tGetKicked — Decision Trees & Ensembles

Бинарная классификация: предсказать проблемную машину (`IsBadBuy`).
Проект про tree-based модели: от собственного CART до LightGBM/CatBoost/XGBoost.
Метрика — **Gini = 2·ROC-AUC − 1**.

## Что сделано

- **Разбиение по `PurchDate`** — train / valid / test по времени
  (по 33%), без shuffle: исключён leakage из будущего.
- **Препроцессинг** — CountEncoder для категорий (фитится только на train,
  новые категории в valid/test → 0), медианная импутация числовых.
- **Собственные реализации с нуля:**
  - `Node` — хранение данных, Gini / std, ссылки на детей
  - `find_best_split` — перебор порогов через сортировку + кумулятивные суммы
  - `find_random_split` — случайный порог на признак (Extra Randomized Tree)
  - `DecisionTreeClassifier` / `DecisionTreeRegressor` (MSE)
  - `RandomForestClassifier` — bagging + random subspace, фиксированный seed
  - `GBDTClassifier` — градиент BCE (`p − y`), инкрементальное обучение,
    `max_depth`, `number_of_trees`, `max_features`, learning rate
  - `ExtraTreesClassifier` — без bootstrap, random splitter
- **Готовые библиотеки** — LightGBM, XGBoost (+ DART), CatBoost.
  Подбор гиперпараметров по valid с early stopping, test не трогался до конца.
- **Анализ различий реализаций** — level-wise vs leaf-wise vs oblivious,
  ordered target statistics в CatBoost, DART-режим в XGBoost.
- **Контроль переобучения** — Gini на train / valid / test.

## Стек

Python 3, numpy, pandas, scikit-learn, category_encoders,
LightGBM, XGBoost, CatBoost.

## Материалы

[Рабочий ноутбук](src/task.ipynb) · [Исходное задание School 21](TASK.md)
