import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import time

wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

print(f"\nРазмер датасета: {X.shape[0]} образцов, {X.shape[1]} признаков")
print(f"Целевые классы: {wine.target_names}")
print(f"Распределение классов:\n{pd.Series(y).value_counts().sort_index()}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Обучающая выборка: {X_train.shape[0]} образцов")
print(f"Тестовая выборка: {X_test.shape[0]} образцов")
print(f"Распределение классов в обучающей:\n{pd.Series(y_train).value_counts().sort_index()}")
print(f"Распределение классов в тестовой:\n{pd.Series(y_test).value_counts().sort_index()}")

results = {}

start_time = time.time()
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=5),
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
bagging.fit(X_train, y_train)
y_pred_bagging = bagging.predict(X_test)
bagging_time = time.time() - start_time

accuracy_bagging = accuracy_score(y_test, y_pred_bagging)
f1_bagging = f1_score(y_test, y_pred_bagging, average='weighted')

print(f"Accuracy: {accuracy_bagging:.4f}")
print(f"F1-score (weighted): {f1_bagging:.4f}")
print(f"Время обучения: {bagging_time:.2f} сек")

results['Bagging'] = {
    'model': bagging,
    'accuracy': accuracy_bagging,
    'f1_score': f1_bagging,
    'time': bagging_time,
    'y_pred': y_pred_bagging
}

start_time = time.time()
random_forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)
rf_time = time.time() - start_time

accuracy_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

print(f"Accuracy: {accuracy_rf:.4f}")
print(f"F1-score (weighted): {f1_rf:.4f}")
print(f"Время обучения: {rf_time:.2f} сек")

results['RandomForest'] = {
    'model': random_forest,
    'accuracy': accuracy_rf,
    'f1_score': f1_rf,
    'time': rf_time,
    'y_pred': y_pred_rf
}

start_time = time.time()
extra_trees = ExtraTreesClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
extra_trees.fit(X_train, y_train)
y_pred_extra = extra_trees.predict(X_test)
extra_time = time.time() - start_time

accuracy_extra = accuracy_score(y_test, y_pred_extra)
f1_extra = f1_score(y_test, y_pred_extra, average='weighted')

print(f"Accuracy: {accuracy_extra:.4f}")
print(f"F1-score (weighted): {f1_extra:.4f}")
print(f"Время обучения: {extra_time:.2f} сек")

results['ExtraTrees'] = {
    'model': extra_trees,
    'accuracy': accuracy_extra,
    'f1_score': f1_extra,
    'time': extra_time,
    'y_pred': y_pred_extra
}

start_time = time.time()
adaboost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    random_state=42
)
adaboost.fit(X_train, y_train)
y_pred_adaboost = adaboost.predict(X_test)
adaboost_time = time.time() - start_time

accuracy_adaboost = accuracy_score(y_test, y_pred_adaboost)
f1_adaboost = f1_score(y_test, y_pred_adaboost, average='weighted')

print(f"Accuracy: {accuracy_adaboost:.4f}")
print(f"F1-score (weighted): {f1_adaboost:.4f}")
print(f"Время обучения: {adaboost_time:.2f} сек")

results['AdaBoost'] = {
    'model': adaboost,
    'accuracy': accuracy_adaboost,
    'f1_score': f1_adaboost,
    'time': adaboost_time,
    'y_pred': y_pred_adaboost
}

start_time = time.time()
gradient_boost = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)
gradient_boost.fit(X_train, y_train)
y_pred_gb = gradient_boost.predict(X_test)
gb_time = time.time() - start_time

accuracy_gb = accuracy_score(y_test, y_pred_gb)
f1_gb = f1_score(y_test, y_pred_gb, average='weighted')

print(f"Accuracy: {accuracy_gb:.4f}")
print(f"F1-score (weighted): {f1_gb:.4f}")
print(f"Время обучения: {gb_time:.2f} сек")

results['GradientBoosting'] = {
    'model': gradient_boost,
    'accuracy': accuracy_gb,
    'f1_score': f1_gb,
    'time': gb_time,
    'y_pred': y_pred_gb
}

comparison = pd.DataFrame({
    'Модель': ['Bagging', 'RandomForest', 'ExtraTrees', 'AdaBoost', 'GradientBoosting'],
    'Accuracy': [accuracy_bagging, accuracy_rf, accuracy_extra, accuracy_adaboost, accuracy_gb],
    'F1-score (weighted)': [f1_bagging, f1_rf, f1_extra, f1_adaboost, f1_gb],
    'Время (сек)': [bagging_time, rf_time, extra_time, adaboost_time, gb_time]
})

print("\n", comparison.round(4))

best_accuracy = comparison.loc[comparison['Accuracy'].idxmax(), 'Модель']
best_f1 = comparison.loc[comparison['F1-score (weighted)'].idxmax(), 'Модель']
fastest = comparison.loc[comparison['Время (сек)'].idxmin(), 'Модель']

print(f"Лучшая модель по Accuracy: {best_accuracy} ({comparison['Accuracy'].max():.4f})")
print(f"Лучшая модель по F1-score: {best_f1} ({comparison['F1-score (weighted)'].max():.4f})")
print(f"Самая быстрая модель: {fastest} ({comparison['Время (сек)'].min():.2f} сек)")

best_model_name = comparison.loc[comparison['Accuracy'].idxmax(), 'Модель']
best_model = results[best_model_name]['model']
print("\nClassification Report:")
print(classification_report(y_test, results[best_model_name]['y_pred'], target_names=wine.target_names))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

x = np.arange(len(comparison['Модель']))
width = 0.35

axes[0, 0].bar(x - width/2, comparison['Accuracy'], width, label='Accuracy', color='skyblue')
axes[0, 0].bar(x + width/2, comparison['F1-score (weighted)'], width, label='F1-score', color='lightcoral')
axes[0, 0].set_xlabel('Модели')
axes[0, 0].set_ylabel('Значение')
axes[0, 0].set_title('Сравнение метрик качества моделей')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(comparison['Модель'], rotation=45, ha='right')
axes[0, 0].legend()
axes[0, 0].set_ylim(0.8, 1.05)
axes[0, 0].grid(True, alpha=0.3)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
axes[0, 1].bar(comparison['Модель'], comparison['Время (сек)'], color=colors)
axes[0, 1].set_xlabel('Модели')
axes[0, 1].set_ylabel('Время (сек)')
axes[0, 1].set_title('Время обучения моделей')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3)

feature_importance = pd.DataFrame({
    'Признак': wine.feature_names,
    'Важность': random_forest.feature_importances_
}).sort_values('Важность', ascending=True)

axes[1, 0].barh(feature_importance['Признак'], feature_importance['Важность'], color='green')
axes[1, 0].set_xlabel('Важность')
axes[1, 0].set_ylabel('Признаки')
axes[1, 0].set_title('Важность признаков (RandomForest)')
axes[1, 0].tick_params(axis='y', labelsize=8)

feature_importance_gb = pd.DataFrame({
    'Признак': wine.feature_names,
    'Важность': gradient_boost.feature_importances_
}).sort_values('Важность', ascending=True)

axes[1, 1].barh(feature_importance_gb['Признак'], feature_importance_gb['Важность'], color='orange')
axes[1, 1].set_xlabel('Важность')
axes[1, 1].set_ylabel('Признаки')
axes[1, 1].set_title('Важность признаков (GradientBoosting)')
axes[1, 1].tick_params(axis='y', labelsize=8)

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(12, 6))

metrics_to_plot = ['Accuracy', 'F1-score (weighted)']
models = comparison['Модель'].tolist()

x = np.arange(len(models))
width = 0.35

for i, metric in enumerate(metrics_to_plot):
    values = comparison[metric].tolist()
    offset = -width/2 if i == 0 else width/2
    bars = ax.bar(x + offset, values, width, label=metric)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Модели', fontsize=12)
ax.set_ylabel('Значение', fontsize=12)
ax.set_title('Сравнение ансамблевых моделей на датасете Wine', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend(loc='lower right')
ax.set_ylim(0.9, 1.05)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()