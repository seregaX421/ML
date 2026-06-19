import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

# Загрузка данных
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

print(f"\nРазмер датасета: {X.shape[0]} образцов, {X.shape[1]} признаков")
print(f"Целевые классы: {wine.target_names}")
print(f"Распределение классов:\n{pd.Series(y).value_counts().sort_index()}")

# Предварительная обработка (масштабирование)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение на обучающую и тестовую выборки

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Обучающая выборка: {X_train.shape[0]} образцов")
print(f"Тестовая выборка: {X_test.shape[0]} образцов")
print(f"Распределение классов в обучающей:\n{pd.Series(y_train).value_counts().sort_index()}")
print(f"Распределение классов в тестовой:\n{pd.Series(y_test).value_counts().sort_index()}")

# Обучение моделей
# Модель 1: Логистическая регрессия

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

accuracy_log = accuracy_score(y_test, y_pred_log)
f1_log = f1_score(y_test, y_pred_log, average='macro')
precision_log = precision_score(y_test, y_pred_log, average='macro')
recall_log = recall_score(y_test, y_pred_log, average='macro')

print(f"Accuracy: {accuracy_log:.4f}")
print(f"Precision (macro): {precision_log:.4f}")
print(f"Recall (macro): {recall_log:.4f}")
print(f"F1-score (macro): {f1_log:.4f}")

# Модель 2: SVM

svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm, average='macro')
precision_svm = precision_score(y_test, y_pred_svm, average='macro')
recall_svm = recall_score(y_test, y_pred_svm, average='macro')

print(f"Accuracy: {accuracy_svm:.4f}")
print(f"Precision (macro): {precision_svm:.4f}")
print(f"Recall (macro): {recall_svm:.4f}")
print(f"F1-score (macro): {f1_svm:.4f}")

# Модель 3: Дерево решений

tree = DecisionTreeClassifier(random_state=42, max_depth=5)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

accuracy_tree = accuracy_score(y_test, y_pred_tree)
f1_tree = f1_score(y_test, y_pred_tree, average='macro')
precision_tree = precision_score(y_test, y_pred_tree, average='macro')
recall_tree = recall_score(y_test, y_pred_tree, average='macro')

print(f"Accuracy: {accuracy_tree:.4f}")
print(f"Precision (macro): {precision_tree:.4f}")
print(f"Recall (macro): {recall_tree:.4f}")
print(f"F1-score (macro): {f1_tree:.4f}")

# Сравнение моделей
comparison = pd.DataFrame({
    'Модель': ['Логистическая регрессия', 'SVM', 'Дерево решений'],
    'Accuracy': [accuracy_log, accuracy_svm, accuracy_tree],
    'Precision (macro)': [precision_log, precision_svm, precision_tree],
    'Recall (macro)': [recall_log, recall_svm, recall_tree],
    'F1-score (macro)': [f1_log, f1_svm, f1_tree]
})

print("\n", comparison.round(4))

best_model = comparison.loc[comparison['Accuracy'].idxmax(), 'Модель']
print(f"Лучшая модель по Accuracy: {best_model} ({comparison['Accuracy'].max():.4f})")

if accuracy_log > accuracy_svm and accuracy_log > accuracy_tree:
    print("Логистическая регрессия показала наилучший результат")
elif accuracy_svm > accuracy_log and accuracy_svm > accuracy_tree:
    print("SVM показал наилучший результат")
else:
    print("Дерево решений показало наилучший результат")

# Важность признаков в дереве решений
feature_importance = pd.DataFrame({
    'Признак': wine.feature_names,
    'Важность': tree.feature_importances_
}).sort_values('Важность', ascending=False)

print("\nВажность признаков:")
print(feature_importance)

# График важности признаков
plt.figure(figsize=(12, 6))
plt.barh(feature_importance['Признак'], feature_importance['Важность'], color='steelblue')
plt.xlabel('Важность', fontsize=12)
plt.ylabel('Признаки', fontsize=12)
plt.title('Важность признаков в дереве решений', fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Визуализация дерева решений
# Графическое представление дерева
plt.figure(figsize=(20, 12))
plot_tree(
    tree,
    feature_names=wine.feature_names,
    class_names=wine.target_names,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title('Дерево решений для классификации вин', fontsize=16)
plt.tight_layout()
plt.show()

# Дополнительная визуализация: матрицы ошибок

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

models = [
    (log_reg, y_pred_log, 'Логистическая регрессия'),
    (svm, y_pred_svm, 'SVM'),
    (tree, y_pred_tree, 'Дерево решений')
]

for idx, (model, y_pred, name) in enumerate(models):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
    axes[idx].set_xlabel('Предсказанные классы')
    axes[idx].set_ylabel('Истинные классы')
    axes[idx].set_title(f'{name}\nAccuracy: {accuracy_score(y_test, y_pred):.4f}')

plt.tight_layout()
plt.show()