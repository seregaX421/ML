import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
import time

wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

print(f"\nРазмер датасета: {X.shape[0]} образцов, {X.shape[1]} признаков")
print(f"Целевые классы: {wine.target_names}")
print(f"Распределение классов:\n{pd.Series(y).value_counts().sort_index()}")
print("\nВНИМАНИЕ: Библиотека gmdh не поддерживает классификацию")
print("Будут продемонстрированы стекинг и многослойный персептрон")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nОбучающая выборка: {X_train.shape[0]} образцов")
print(f"Тестовая выборка: {X_test.shape[0]} образцов")

print("МОДЕЛЬ СТЕКИНГА (STACKING)")

base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42))
]

meta_learner = LogisticRegression(max_iter=1000, random_state=42)

start_time = time.time()
stacking = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,
    stack_method='predict_proba'
)
stacking.fit(X_train, y_train)
y_pred_stacking = stacking.predict(X_test)
stacking_time = time.time() - start_time

accuracy_stacking = accuracy_score(y_test, y_pred_stacking)
f1_stacking = f1_score(y_test, y_pred_stacking, average='weighted')

print(f"\nРезультаты стекинга:")
print(f"Базовые модели: RandomForest, GradientBoosting")
print(f"Мета-модель: LogisticRegression")
print(f"Accuracy: {accuracy_stacking:.4f}")
print(f"F1-score (weighted): {f1_stacking:.4f}")
print(f"Время обучения: {stacking_time:.2f} сек")

print("\nClassification Report (Stacking):")
print(classification_report(y_test, y_pred_stacking, target_names=wine.target_names))

print("МНОГОСЛОЙНЫЙ ПЕРСЕПТРОН (MLP)")

start_time = time.time()
mlp_sklearn = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)
mlp_sklearn.fit(X_train, y_train)
y_pred_mlp = mlp_sklearn.predict(X_test)
mlp_time = time.time() - start_time

accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
f1_mlp = f1_score(y_test, y_pred_mlp, average='weighted')

print(f"Архитектура: 13 - 100 - 50 - 3")
print(f"Функция активации: ReLU")
print(f"Оптимизатор: Adam")
print(f"Accuracy: {accuracy_mlp:.4f}")
print(f"F1-score (weighted): {f1_mlp:.4f}")
print(f"Время обучения: {mlp_time:.2f} сек")

print("\nClassification Report (MLP sklearn):")
print(classification_report(y_test, y_pred_mlp, target_names=wine.target_names))

try:
    print("МЕТОДЫ МГУА")
    
    from gmdh import GMDH
    from gmdh import Regressor
    
    print("\nБиблиотека gmdh загружена, но не поддерживает классификацию")
    
    from sklearn.metrics import r2_score, mean_squared_error
    
    y_reg_train = y_train.astype(float)
    y_reg_test = y_test.astype(float)
    
    gmdh_linear = Regressor(
        criterion='mse',
        max_layer=5,
        ref_terms=2
    )
    
    start_time = time.time()
    gmdh_linear.fit(X_train, y_reg_train)
    y_pred_gmdh = gmdh_linear.predict(X_test)
    gmdh_time = time.time() - start_time
    
    r2 = r2_score(y_reg_test, y_pred_gmdh)
    mse = mean_squared_error(y_reg_test, y_pred_gmdh)
    
    print(f"\nРезультаты GMDH (режим регрессии):")
    print(f"R² score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"Время обучения: {gmdh_time:.2f} сек")
    
    y_pred_class = np.round(y_pred_gmdh).clip(0, 2).astype(int)
    accuracy_gmdh = accuracy_score(y_test, y_pred_class)
    print(f"Accuracy (после округления до классов): {accuracy_gmdh:.4f}")
    
except ImportError:
    print("МЕТОДЫ МГУА")
    accuracy_gmdh = None
    gmdh_time = None

print("СРАВНЕНИЕ МОДЕЛЕЙ")

comparison_data = {
    'Модель': ['Stacking (RF+GB+LR)', 'MLP (100-50)'],
    'Accuracy': [accuracy_stacking, accuracy_mlp],
    'F1-score (weighted)': [f1_stacking, f1_mlp],
    'Время (сек)': [stacking_time, mlp_time]
}

if accuracy_gmdh is not None:
    comparison_data['Модель'].append('GMDH (регрессия+округление)')
    comparison_data['Accuracy'].append(accuracy_gmdh)
    comparison_data['F1-score (weighted)'].append(None)
    comparison_data['Время (сек)'].append(gmdh_time)

comparison = pd.DataFrame(comparison_data)
print("\n", comparison.round(4))

best_accuracy = comparison.loc[comparison['Accuracy'].idxmax(), 'Модель']
best_f1 = comparison.loc[comparison['F1-score (weighted)'].idxmax(), 'Модель']
fastest = comparison.loc[comparison['Время (сек)'].idxmin(), 'Модель']

print(f"\nЛучшая модель по Accuracy: {best_accuracy} ({comparison['Accuracy'].max():.4f})")
if best_f1:
    print(f"Лучшая модель по F1-score: {best_f1} ({comparison['F1-score (weighted)'].max():.4f})")
print(f"Самая быстрая модель: {fastest} ({comparison['Время (сек)'].min():.2f} сек)")

print("ВИЗУАЛИЗАЦИЯ")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

x = np.arange(len(comparison['Модель']))
width = 0.35

axes[0, 0].bar(x, comparison['Accuracy'], width, label='Accuracy', color='skyblue')
if 'F1-score (weighted)' in comparison.columns and comparison['F1-score (weighted)'].notna().any():
    axes[0, 0].bar(x + width, comparison['F1-score (weighted)'], width, label='F1-score', color='lightcoral')
axes[0, 0].set_xlabel('Модели')
axes[0, 0].set_ylabel('Значение')
axes[0, 0].set_title('Сравнение метрик качества')
axes[0, 0].set_xticks(x if 'F1-score' not in comparison.columns else x + width/2)
axes[0, 0].set_xticklabels(comparison['Модель'], rotation=15, ha='right')
axes[0, 0].legend()
axes[0, 0].set_ylim(0.7, 1.05)
axes[0, 0].grid(True, alpha=0.3)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
axes[0, 1].bar(comparison['Модель'], comparison['Время (сек)'], color=colors[:len(comparison)])
axes[0, 1].set_xlabel('Модели')
axes[0, 1].set_ylabel('Время (сек)')
axes[0, 1].set_title('Время обучения моделей')
axes[0, 1].tick_params(axis='x', rotation=15)
axes[0, 1].grid(True, alpha=0.3)

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm_stacking = confusion_matrix(y_test, y_pred_stacking)
sns.heatmap(cm_stacking, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_xlabel('Предсказанные классы')
axes[1, 0].set_ylabel('Истинные классы')
axes[1, 0].set_title('Матрица ошибок (Stacking)')

cm_mlp = confusion_matrix(y_test, y_pred_mlp)
sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Greens', ax=axes[1, 1])
axes[1, 1].set_xlabel('Предсказанные классы')
axes[1, 1].set_ylabel('Истинные классы')
axes[1, 1].set_title('Матрица ошибок (MLP)')

plt.tight_layout()
plt.show()

print("АНАЛИЗ ПРОГНОЗОВ МОДЕЛЕЙ")

comparison_predictions = pd.DataFrame({
    'Истинный класс': y_test,
    'Stacking': y_pred_stacking,
    'MLP': y_pred_mlp
})

print("\nСравнение предсказаний (первые 20 тестовых образцов):")
print(comparison_predictions.head(20))

agreement = (y_pred_stacking == y_pred_mlp).mean()
print(f"\nСогласованность моделей (совпадение предсказаний): {agreement:.2%}")

both_correct = ((y_pred_stacking == y_test) & (y_pred_mlp == y_test)).mean()
print(f"Обе модели правильные: {both_correct:.2%}")

stacking_only_correct = ((y_pred_stacking == y_test) & (y_pred_mlp != y_test)).mean()
print(f"Только Stacking правильный: {stacking_only_correct:.2%}")

mlp_only_correct = ((y_pred_mlp == y_test) & (y_pred_stacking != y_test)).mean()
print(f"Только MLP правильный: {mlp_only_correct:.2%}")