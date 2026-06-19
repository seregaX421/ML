import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target_name'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("КОРРЕЛЯЦИОННЫЙ АНАЛИЗ ДАТАСЕТА IRIS")

print(f"\nРазмер датасета: {df.shape[0]} строк, {df.shape[1]} колонок")
print(f"Признаки: {iris.feature_names}")
print(f"Целевые классы: {iris.target_names}")

print("\nПроверка наличия пропусков:")
print(df.isnull().sum())

print("\nПервые 5 строк данных:")
print(df.head())

corr_matrix = df[iris.feature_names].corr()

print("\nКОРРЕЛЯЦИОННАЯ МАТРИЦА")
print(corr_matrix.round(3))

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Тепловая карта корреляции признаков Iris', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

corr_with_target = df[iris.feature_names].corrwith(df['target']).sort_values(ascending=False)

print("\nКОРРЕЛЯЦИЯ ПРИЗНАКОВ С ЦЕЛЕВОЙ ПЕРЕМЕННОЙ")
print(corr_with_target.round(3))

print("ВЫВОДЫ НА ОСНОВЕ РЕЗУЛЬТАТОВ АНАЛИЗА")

high_corr_pairs = []
for i in range(len(iris.feature_names)):
    for j in range(i+1, len(iris.feature_names)):
        corr = corr_matrix.iloc[i, j]
        if abs(corr) > 0.7:
            high_corr_pairs.append((iris.feature_names[i], iris.feature_names[j], corr))

if high_corr_pairs:
    print("\n1. ОБНАРУЖЕНЫ СИЛЬНЫЕ КОРРЕЛЯЦИИ МЕЖДУ ПРИЗНАКАМИ (|r| > 0.7):")
    for f1, f2, corr in high_corr_pairs:
        print(f"   - {f1} и {f2}: {corr:.3f}")
    print("   → Это может привести к мультиколлинеарности в линейных моделях")
else:
    print("\n1. СИЛЬНЫХ КОРРЕЛЯЦИЙ МЕЖДУ ПРИЗНАКАМИ НЕ ОБНАРУЖЕНО")

print("\n2. АНАЛИЗ КОРРЕЛЯЦИИ С ЦЕЛЕВОЙ ПЕРЕМЕННОЙ:")
for feature, corr in corr_with_target.items():
    if corr > 0.7:
        print(f"   - {feature}: {corr:.3f} (очень сильная положительная)")
    elif corr > 0.5:
        print(f"   - {feature}: {corr:.3f} (средняя положительная)")
    elif corr < -0.5:
        print(f"   - {feature}: {corr:.3f} (средняя отрицательная)")
    else:
        print(f"   - {feature}: {corr:.3f} (слабая)")

best_feature = corr_with_target.index[0]
best_corr = corr_with_target.values[0]
print(f"\nНаиболее информативный признак: {best_feature} (r = {best_corr:.3f})")

plt.figure(figsize=(8, 5))
corr_with_target.plot(kind='bar', color='steelblue', edgecolor='black')
plt.xlabel('Признаки', fontsize=12)
plt.ylabel('Корреляция с целевой переменной', fontsize=12)
plt.title('Вклад признаков в предсказание класса Iris', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n3. ВОЗМОЖНОСТЬ ПОСТРОЕНИЯ МОДЕЛЕЙ МАШИННОГО ОБУЧЕНИЯ:")
if high_corr_pairs:
    print("Использования регуляризации (Ridge, Lasso)")
    print("Применения PCA для уменьшения размерности")
    print("Использования нелинейных моделей (деревья решений, ансамбли)")
else:
    print("Мультиколлинеарность не является проблемой для данного набора данных")

print("\n4. ВОЗМОЖНЫЙ ВКЛАД ПРИЗНАКОВ В МОДЕЛЬ:")
for feature, corr in corr_with_target.items():
    if abs(corr) > 0.7:
        print(f"   - {feature}: высокий вклад (|r| = {abs(corr):.3f})")
    elif abs(corr) > 0.4:
        print(f"   - {feature}: средний вклад (|r| = {abs(corr):.3f})")
    else:
        print(f"   - {feature}: низкий вклад (|r| = {abs(corr):.3f})")