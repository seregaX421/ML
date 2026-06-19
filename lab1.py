import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

sns.set(style="ticks")

# Настройка отображения pandas
pd.set_option('display.max_columns', None)

# Загрузка датасета Wine
wine_data = load_wine()

# Создание DataFrame с признаками
data = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)

# Добавление целевого признака
data['target'] = wine_data.target

# Информация о датасете
print("\nДатасет Wine содержит результаты химического анализа вин из Италии.")
print(f"Количество образцов: {data.shape[0]}")
print(f"Количество признаков: {data.shape[1] - 1}")
print(f"Количество классов: 3 (class_0, class_1, class_2)")
print(f"\nСписок признаков (химические компоненты):")
for i, feature in enumerate(wine_data.feature_names, 1):
    print(f"  {i}. {feature}")
print(f"\nЦелевой признак: target (0, 1, 2) - три различных сорта вин")

print("\n\n2) ОСНОВНЫЕ ХАРАКТЕРИСТИКИ ДАТАСЕТА")

# Первые 5 строк
print("\nПервые 5 строк датасета:")
print(data.head())

# Размер датасета
print(f"\nРазмер датасета: {data.shape[0]} строк, {data.shape[1]} колонок")

# Список колонок
print(f"\nСписок колонок: {data.columns.tolist()}")

# Типы данных
print("\nТипы данных колонок:")
print(data.dtypes)

# Проверка наличия пустых значений
print("\nПроверка наличия пустых значений:")
for col in data.columns:
    null_count = data[col].isnull().sum()
    print(f"  {col}: {null_count} пропущенных значений")

# Основные статистические характеристики
print("\nОсновные статистические характеристики набора данных:")
print(data.describe())

# Уникальные значения целевого признака
print("\nУникальные значения целевого признака (target):")
print(f"Значения: {data['target'].unique()}")
print(f"Распределение классов:")
class_counts = data['target'].value_counts().sort_index()
for i, count in class_counts.items():
    print(f"  Класс {i}: {count} образцов ({count/len(data)*100:.1f}%)")

# Создаем фигуру для нескольких графиков
fig = plt.figure(figsize=(16, 20))

# Диаграмма рассеяния
print("\nПостроение диаграммы рассеяния (Alcohol vs Flavanoids)...")
ax1 = fig.add_subplot(3, 3, 1)
scatter = ax1.scatter(data['alcohol'], data['flavanoids'], 
                       c=data['target'], cmap='viridis', alpha=0.7, edgecolors='black')
ax1.set_xlabel('Alcohol', fontsize=12)
ax1.set_ylabel('Flavanoids', fontsize=12)
ax1.set_title('Диаграмма рассеяния: Alcohol vs Flavanoids\n(цвет по классам вина)', fontsize=14)
plt.colorbar(scatter, ax=ax1, label='Class')

# Вторая диаграмма рассеяния
ax2 = fig.add_subplot(3, 3, 2)
scatter2 = ax2.scatter(data['color_intensity'], data['proline'], 
                        c=data['target'], cmap='plasma', alpha=0.7, edgecolors='black')
ax2.set_xlabel('Color Intensity', fontsize=12)
ax2.set_ylabel('Proline', fontsize=12)
ax2.set_title('Диаграмма рассеяния: Color Intensity vs Proline', fontsize=14)
plt.colorbar(scatter2, ax=ax2, label='Class')

# Гистограмма распределения Alcohol по классам
ax3 = fig.add_subplot(3, 3, 3)
for target_val in [0, 1, 2]:
    subset = data[data['target'] == target_val]
    ax3.hist(subset['alcohol'], bins=15, alpha=0.5, label=f'Class {target_val}')
ax3.set_xlabel('Alcohol', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.set_title('Гистограмма: Распределение Alcohol по классам', fontsize=14)
ax3.legend()

# Гистограмма распределения Flavanoids
ax4 = fig.add_subplot(3, 3, 4)
sns.histplot(data=data, x='flavanoids', hue='target', multiple='stack', bins=20, ax=ax4)
ax4.set_xlabel('Flavanoids', fontsize=12)
ax4.set_ylabel('Density', fontsize=12)
ax4.set_title('Гистограмма: Распределение Flavanoids', fontsize=14)

# Ящик с усами (Boxplot) для Alcohol
ax5 = fig.add_subplot(3, 3, 5)
sns.boxplot(x='target', y='alcohol', data=data, ax=ax5)
ax5.set_xlabel('Target Class', fontsize=12)
ax5.set_ylabel('Alcohol', fontsize=12)
ax5.set_title('Boxplot: Alcohol по классам вин', fontsize=14)

# Ящик с усами для Proline
ax6 = fig.add_subplot(3, 3, 6)
sns.boxplot(x='target', y='proline', data=data, ax=ax6)
ax6.set_xlabel('Target Class', fontsize=12)
ax6.set_ylabel('Proline', fontsize=12)
ax6.set_title('Boxplot: Proline по классам вин', fontsize=14)

# Violin plot (альтернатива boxplot)
ax7 = fig.add_subplot(3, 3, 7)
sns.violinplot(x='target', y='color_intensity', data=data, ax=ax7)
ax7.set_xlabel('Target Class', fontsize=12)
ax7.set_ylabel('Color Intensity', fontsize=12)
ax7.set_title('Violin plot: Color Intensity по классам', fontsize=14)

# Pairplot для нескольких признаков (используем отдельную фигуру)
plt.tight_layout()
plt.show()

# Отдельный pairplot для выбранных признаков
print("\nПостроение pairplot для выбранных признаков...")
selected_features = ['alcohol', 'flavanoids', 'color_intensity', 'proline']
pairplot_fig = sns.pairplot(data[selected_features + ['target']], 
                             hue='target', 
                             diag_kind='hist',
                             palette='Set1',
                             plot_kws={'alpha': 0.6})
pairplot_fig.fig.suptitle('Pairplot: Взаимосвязи между основными признаками', y=1.02, fontsize=16)
plt.show()

# Вычисление корреляционной матрицы
corr_matrix = data.drop('target', axis=1).corr()
print("\nКорреляционная матрица (первые 10x10):")
print(corr_matrix.iloc[:10, :10].round(3))

# Корреляция с целевым признаком
print("\nКорреляция признаков с целевым признаком (target):")
target_corr = data.drop('target', axis=1).corrwith(data['target']).sort_values(ascending=False)
for feature, corr_value in target_corr.items():
    print(f"  {feature}: {corr_value:.4f}")

# Вывод наиболее и наименее коррелирующих признаков
print("\nНаиболее коррелирующие признаки с целевым признаком (топ-5):")
for feature, corr_value in target_corr.head(5).items():
    print(f"  ↑ {feature}: {corr_value:.4f}")

print("\nНаименее коррелирующие признаки с целевым признаком (топ-5):")
for feature, corr_value in target_corr.tail(5).items():
    print(f"  ↓ {feature}: {corr_value:.4f}")

# Тепловая карта корреляционной матрицы (Heatmap)
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, 
            annot=True, 
            fmt='.2f', 
            cmap='RdBu_r',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8})
plt.title('Тепловая карта корреляционной матрицы признаков Wine dataset', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Уменьшенная тепловая карта только для наиболее коррелирующих признаков
top_features = target_corr.head(8).index.tolist()
corr_top = data[top_features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_top, 
            annot=True, 
            fmt='.3f', 
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8})
plt.title('Корреляционная матрица топ-8 признаков, наиболее коррелирующих с target', fontsize=14)
plt.tight_layout()
plt.show()

# Распределение всех признаков
fig, axes = plt.subplots(4, 4, figsize=(16, 14))
axes = axes.flatten()

for i, col in enumerate(data.drop('target', axis=1).columns):
    axes[i].hist(data[col], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    axes[i].set_xlabel(col, fontsize=10)
    axes[i].set_ylabel('Frequency', fontsize=10)
    axes[i].set_title(f'Distribution of {col}', fontsize=11)

# Убираем пустые подграфики
for j in range(len(data.drop('target', axis=1).columns), len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Гистограммы распределения всех признаков Wine dataset', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# Распределение целевой переменной
print("\nРаспределение целевой переменной:")
plt.figure(figsize=(8, 5))
colors = ['#ff9999', '#66b3ff', '#99ff99']
plt.pie(class_counts.values, labels=[f'Class {i}' for i in class_counts.index], 
        autopct='%1.1f%%', colors=colors, startangle=90, explode=(0.05, 0.05, 0.05))
plt.title('Распределение классов вин в датасете', fontsize=14)
plt.show()