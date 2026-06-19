import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Загрузка датасета Wine
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

print(f"\nРазмер датасета: {X.shape[0]} образцов, {X.shape[1]} признаков")
print(f"Целевые классы: {wine.target_names}")
print(f"Распределение классов:\n{pd.Series(y).value_counts().sort_index()}")

# Предварительная обработка данных (масштабирование)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Выполнено масштабирование признаков (StandardScaler)")
print(f"Размер признакового пространства: {X_scaled.shape}")

# Разделение на обучающую и тестовую выборки

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Обучающая выборка: {X_train.shape[0]} образцов")
print(f"Тестовая выборка: {X_test.shape[0]} образцов")
print(f"Распределение классов в обучающей выборке:\n{pd.Series(y_train).value_counts().sort_index()}")
print(f"Распределение классов в тестовой выборке:\n{pd.Series(y_test).value_counts().sort_index()}")

# Модель ближайших соседей с произвольным K
knn_default = KNeighborsClassifier(n_neighbors=5)
knn_default.fit(X_train, y_train)
y_pred_default = knn_default.predict(X_test)

# Оценка качества модели с произвольным K
print("\nМетрики качества для K = 5:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_default):.4f}")
print(f"Precision (macro): {precision_score(y_test, y_pred_default, average='macro'):.4f}")
print(f"Recall (macro): {recall_score(y_test, y_pred_default, average='macro'):.4f}")
print(f"F1-score (macro): {f1_score(y_test, y_pred_default, average='macro'):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_default, target_names=wine.target_names))

# Кросс-валидация для модели с K=5
cv_scores_default = cross_val_score(knn_default, X_train, y_train, cv=5)
print(f"\nКросс-валидация (5-fold) для K=5:")
print(f"Средняя точность: {cv_scores_default.mean():.4f} (+/- {cv_scores_default.std() * 2:.4f})")

# Подбор гиперпараметра K с использованием GridSearchCV
param_grid = {'n_neighbors': range(1, 31)}

# Стратегия 1: K-Fold кросс-валидация
kf = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=kf,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"\nЛучший параметр K (GridSearchCV): {grid_search.best_params_['n_neighbors']}")
print(f"Лучшая точность на кросс-валидации: {grid_search.best_score_:.4f}")

# Оценка лучшей модели на тестовой выборке
best_knn_grid = grid_search.best_estimator_
y_pred_grid = best_knn_grid.predict(X_test)

print("\nМетрики качества для оптимальной модели (GridSearchCV):")
print(f"Accuracy: {accuracy_score(y_test, y_pred_grid):.4f}")
print(f"Precision (macro): {precision_score(y_test, y_pred_grid, average='macro'):.4f}")
print(f"Recall (macro): {recall_score(y_test, y_pred_grid, average='macro'):.4f}")
print(f"F1-score (macro): {f1_score(y_test, y_pred_grid, average='macro'):.4f}")

# Подбор гиперпараметра K с использованием RandomizedSearchCV
param_dist = {'n_neighbors': range(1, 51)}

# Стратегия 2: Stratified K-Fold кросс-валидация
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
random_search = RandomizedSearchCV(
    KNeighborsClassifier(),
    param_dist,
    n_iter=30,
    cv=skf,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, y_train)

print(f"\nЛучший параметр K (RandomizedSearchCV): {random_search.best_params_['n_neighbors']}")
print(f"Лучшая точность на кросс-валидации: {random_search.best_score_:.4f}")

# Оценка лучшей модели на тестовой выборке
best_knn_random = random_search.best_estimator_
y_pred_random = best_knn_random.predict(X_test)

print("\nМетрики качества для оптимальной модели (RandomizedSearchCV):")
print(f"Accuracy: {accuracy_score(y_test, y_pred_random):.4f}")
print(f"Precision (macro): {precision_score(y_test, y_pred_random, average='macro'):.4f}")
print(f"Recall (macro): {recall_score(y_test, y_pred_random, average='macro'):.4f}")
print(f"F1-score (macro): {f1_score(y_test, y_pred_random, average='macro'):.4f}")

# Сравнение метрик
results = pd.DataFrame({
    'Модель': ['KNN (K=5)', 'KNN (GridSearchCV)', 'KNN (RandomizedSearchCV)'],
    'Best_K': [5, grid_search.best_params_['n_neighbors'], random_search.best_params_['n_neighbors']],
    'Accuracy': [accuracy_score(y_test, y_pred_default),
                 accuracy_score(y_test, y_pred_grid),
                 accuracy_score(y_test, y_pred_random)],
    'Precision (macro)': [precision_score(y_test, y_pred_default, average='macro'),
                          precision_score(y_test, y_pred_grid, average='macro'),
                          precision_score(y_test, y_pred_random, average='macro')],
    'Recall (macro)': [recall_score(y_test, y_pred_default, average='macro'),
                       recall_score(y_test, y_pred_grid, average='macro'),
                       recall_score(y_test, y_pred_random, average='macro')],
    'F1-score (macro)': [f1_score(y_test, y_pred_default, average='macro'),
                         f1_score(y_test, y_pred_grid, average='macro'),
                         f1_score(y_test, y_pred_random, average='macro')]
})

print("\n", results.round(4))

print(f"1. Модель с произвольным K=5 показала точность: {accuracy_score(y_test, y_pred_default):.4f}")
print(f"2. GridSearchCV нашел оптимальный K = {grid_search.best_params_['n_neighbors']} с точностью: {accuracy_score(y_test, y_pred_grid):.4f}")
print(f"3. RandomizedSearchCV нашел оптимальный K = {random_search.best_params_['n_neighbors']} с точностью: {accuracy_score(y_test, y_pred_random):.4f}")

if accuracy_score(y_test, y_pred_grid) > accuracy_score(y_test, y_pred_default):
    improvement = accuracy_score(y_test, y_pred_grid) - accuracy_score(y_test, y_pred_default)
    print(f"4. Улучшение качества после подбора гиперпараметра: {improvement:.4f} (+{improvement * 100:.2f}%)")
else:
    print("4. Модель с произвольным K показала результат, близкий к оптимальному")
