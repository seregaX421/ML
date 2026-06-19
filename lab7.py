import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

# Загружаем датасет Wine
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

print(f"\n📊 Информация о датасете:")
print(f"   Количество образцов: {len(df)}")
print(f"   Количество признаков: {len(wine.feature_names)}")
print(f"   Целевая переменная: качество вина (0-2)")

df_sorted = df.sort_values('target')
time_series = df_sorted['alcohol'].values

print(f"\n📈 Временной ряд (содержание алкоголя, отсортированный по качеству):")
print(f"   Длина ряда: {len(time_series)}")
print(f"   Min: {time_series.min():.2f}")
print(f"   Max: {time_series.max():.2f}")
print(f"   Mean: {time_series.mean():.2f}")
print(f"   Std: {time_series.std():.2f}")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# График 1: Временной ряд
axes[0, 0].plot(time_series, color='blue', linewidth=1.5)
axes[0, 0].set_title('Временной ряд: Содержание алкоголя в вине', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Время (индекс)')
axes[0, 0].set_ylabel('Алкоголь (%)')
axes[0, 0].grid(True, alpha=0.3)

# График 2: Гистограмма распределения
axes[0, 1].hist(time_series, bins=20, color='green', alpha=0.7, edgecolor='black')
axes[0, 1].set_title('Распределение значений', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Алкоголь (%)')
axes[0, 1].set_ylabel('Частота')
axes[0, 1].grid(True, alpha=0.3)

# График 3: Скользящее среднее
window = 10
rolling_mean = pd.Series(time_series).rolling(window=window).mean()
rolling_std = pd.Series(time_series).rolling(window=window).std()

axes[1, 0].plot(time_series, color='blue', alpha=0.5, label='Исходный ряд')
axes[1, 0].plot(rolling_mean, color='red', linewidth=2, label=f'Скользящее среднее (win={window})')
axes[1, 0].fill_between(range(len(time_series)), 
                       rolling_mean - rolling_std, 
                       rolling_mean + rolling_std, 
                       color='red', alpha=0.1)
axes[1, 0].set_title('Скользящее среднее и стандартное отклонение', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Время (индекс)')
axes[1, 0].set_ylabel('Алкоголь (%)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# График 4: Автокорреляция
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(time_series, ax=axes[1, 1], lags=20)
axes[1, 1].set_title('Автокорреляционная функция (ACF)', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('АНАЛИЗ ВРЕМЕННОГО РЯДА - ДАННЫЕ О ВИНЕ', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Используем 80% для обучения, 20% для теста
train_size = int(len(time_series) * 0.8)
train_data = time_series[:train_size]
test_data = time_series[train_size:]

print(f"\n Разделение на обучающую и тестовую выборку:")
print(f"   Обучающая выборка: {len(train_data)} наблюдений ({len(train_data)/len(time_series)*100:.0f}%)")
print(f"   Тестовая выборка: {len(test_data)} наблюдений ({len(test_data)/len(time_series)*100:.0f}%)")

# Проверка стационарности
result = adfuller(train_data)
print(f"\n📊 Тест Дики-Фуллера на стационарность:")
print(f"   Статистика: {result[0]:.4f}")
print(f"   p-value: {result[1]:.4f}")

# Создаем и обучаем модель ARIMA
# Используем порядок (p=2, d=1, q=2) - простой вариант
model_arima = ARIMA(train_data, order=(2, 1, 2))
model_fit = model_arima.fit()

print(f"\n📈 Модель ARIMA обучена:")
print(f"   AIC: {model_fit.aic:.2f}")
print(f"   BIC: {model_fit.bic:.2f}")

# Прогнозирование
forecast_arima = model_fit.forecast(steps=len(test_data))
forecast_arima = np.array(forecast_arima)

# Простая реализация символьной регрессии через полиномиальную аппроксимацию
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Создаем признаки - время
X_train = np.arange(len(train_data)).reshape(-1, 1)
X_test = np.arange(len(train_data), len(time_series)).reshape(-1, 1)

# Полиномиальная регрессия (как аналог символьной)
poly = PolynomialFeatures(degree=3)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

model_poly = LinearRegression()
model_poly.fit(X_poly_train, train_data)

# Прогноз
forecast_poly = model_poly.predict(X_poly_test)

# Для сравнения также построим простую символьную функцию
# y = a + b*x + c*x^2 + d*x^3
coefs = model_poly.coef_
intercept = model_poly.intercept_

print(f"\n Полученная полиномиальная функция:")
print(f"   f(x) = {intercept:.4f} + {coefs[1]:.4f}x + {coefs[2]:.4f}x² + {coefs[3]:.4f}x³")
print(f"   R² на обучении: {model_poly.score(X_poly_train, train_data):.4f}")

print("\n" + "=" * 80)
print("МЕТОД 3: МГУА - ЛИНЕЙНЫЙ (COMBI)")
print("=" * 80)

# Простая реализация линейного МГУА
def gmdh_combi(X, y, test_X, max_terms=5):
    """
    Линейная комбинация полиномиальных членов
    """
    # Создаем матрицу признаков: x, x², x³, sin(x), cos(x)
    n = len(X)
    features = np.column_stack([
        X.flatten(),
        X.flatten()**2,
        X.flatten()**3,
        np.sin(X.flatten()),
        np.cos(X.flatten())
    ])
    
    # Стандартизация
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Выбираем лучшие признаки по корреляции
    correlations = []
    for i in range(features_scaled.shape[1]):
        corr = np.abs(np.corrcoef(features_scaled[:, i], y)[0, 1])
        correlations.append(corr)
    
    # Берем топ признаков
    top_indices = np.argsort(correlations)[-max_terms:]
    
    # Строим модель на отобранных признаках
    X_selected = features_scaled[:, top_indices]
    
    # Линейная регрессия
    model = LinearRegression()
    model.fit(X_selected, y)
    
    # Прогноз на тесте
    test_features = np.column_stack([
        test_X.flatten(),
        test_X.flatten()**2,
        test_X.flatten()**3,
        np.sin(test_X.flatten()),
        np.cos(test_X.flatten())
    ])
    test_features_scaled = scaler.transform(test_features)
    test_selected = test_features_scaled[:, top_indices]
    forecast = model.predict(test_selected)
    
    return forecast, model, top_indices

print("\n Обучение линейной модели МГУА (COMBI)...")
forecast_combi, model_combi, selected_indices = gmdh_combi(X_train, train_data, X_test)

print(f"\n Модель COMBI обучена:")
print(f"   Отобрано признаков: {len(selected_indices)}")
print(f"   Коэффициенты: {model_combi.coef_}")

def gmdh_mia(X, y, test_X, max_terms=8):
    """
    Нелинейная МГУА с полиномами Колмогорова-Габора
    """
    X_flat = X.flatten()
    test_flat = test_X.flatten()
    
    # Создаем базовые функции
    n = len(X_flat)
    features = []
    feature_names = []
    
    # Полиномы до 2-й степени
    features.append(np.ones(n))
    feature_names.append('1')
    features.append(X_flat)
    feature_names.append('x')
    features.append(X_flat**2)
    feature_names.append('x²')
    features.append(X_flat**3)
    feature_names.append('x³')
    
    # Тригонометрические
    features.append(np.sin(X_flat))
    feature_names.append('sin(x)')
    features.append(np.cos(X_flat))
    feature_names.append('cos(x)')
    
    # Экспоненциальные
    features.append(np.exp(-X_flat/10))
    feature_names.append('exp(-x/10)')
    
    # Комбинации
    features.append(X_flat * np.sin(X_flat))
    feature_names.append('x*sin(x)')
    
    features = np.column_stack(features)
    
    # Стандартизация
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Отбор признаков по корреляции
    correlations = []
    for i in range(features_scaled.shape[1]):
        corr = np.abs(np.corrcoef(features_scaled[:, i], y)[0, 1])
        correlations.append(corr)
    
    top_indices = np.argsort(correlations)[-max_terms:]
    
    # Строим нелинейную модель (полиномиальную)
    X_selected = features_scaled[:, top_indices]
    
    # Добавляем взаимодействия
    interaction_features = []
    for i in range(len(top_indices)):
        for j in range(i+1, len(top_indices)):
            interaction_features.append(X_selected[:, i] * X_selected[:, j])
    
    if interaction_features:
        X_interaction = np.column_stack([X_selected] + interaction_features)
    else:
        X_interaction = X_selected
    
    # Нелинейная регрессия
    model = LinearRegression()
    model.fit(X_interaction, y)
    
    # Прогноз
    test_features = np.column_stack([
        np.ones(len(test_flat)),
        test_flat,
        test_flat**2,
        test_flat**3,
        np.sin(test_flat),
        np.cos(test_flat),
        np.exp(-test_flat/10),
        test_flat * np.sin(test_flat)
    ])
    test_features_scaled = scaler.transform(test_features)
    test_selected = test_features_scaled[:, top_indices]
    
    test_interaction = []
    for i in range(len(top_indices)):
        for j in range(i+1, len(top_indices)):
            test_interaction.append(test_selected[:, i] * test_selected[:, j])
    
    if test_interaction:
        test_final = np.column_stack([test_selected] + test_interaction)
    else:
        test_final = test_selected
    
    forecast = model.predict(test_final)
    
    return forecast, model, top_indices

print("\n Обучение нелинейной модели МГУА (MIA)...")
forecast_mia, model_mia, mia_indices = gmdh_mia(X_train, train_data, X_test)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Общий график
ax = axes[0, 0]
ax.plot(range(len(train_data)), train_data, color='blue', label='Обучающая выборка', linewidth=2)
ax.plot(range(len(train_data), len(time_series)), test_data, color='black', label='Тестовая выборка', linewidth=2)
ax.plot(range(len(train_data), len(time_series)), forecast_arima, color='red', label='ARIMA', linewidth=2, linestyle='--')
ax.plot(range(len(train_data), len(time_series)), forecast_poly, color='green', label='Символьная рег.', linewidth=2, linestyle='--')
ax.plot(range(len(train_data), len(time_series)), forecast_combi, color='orange', label='МГУА (COMBI)', linewidth=2, linestyle='--')
ax.plot(range(len(train_data), len(time_series)), forecast_mia, color='purple', label='МГУА (MIA)', linewidth=2, linestyle='--')
ax.set_title('Сравнение прогнозов', fontsize=12, fontweight='bold')
ax.set_xlabel('Время (индекс)')
ax.set_ylabel('Алкоголь (%)')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# Отдельный график для ARIMA
ax = axes[0, 1]
ax.plot(range(len(train_data), len(time_series)), test_data, color='black', label='Факт', linewidth=2)
ax.plot(range(len(train_data), len(time_series)), forecast_arima, color='red', label='Прогноз ARIMA', linewidth=2)
ax.fill_between(range(len(train_data), len(time_series)), 
                forecast_arima - 0.5, forecast_arima + 0.5, 
                color='red', alpha=0.1)
ax.set_title('Метод 1: ARIMA', fontsize=12, fontweight='bold')
ax.set_xlabel('Время (индекс)')
ax.set_ylabel('Алкоголь (%)')
ax.legend()
ax.grid(True, alpha=0.3)

# Отдельный график для символьной регрессии
ax = axes[1, 0]
ax.plot(range(len(train_data), len(time_series)), test_data, color='black', label='Факт', linewidth=2)
ax.plot(range(len(train_data), len(time_series)), forecast_poly, color='green', label='Символьная рег.', linewidth=2)
ax.set_title('Метод 2: Символьная регрессия', fontsize=12, fontweight='bold')
ax.set_xlabel('Время (индекс)')
ax.set_ylabel('Алкоголь (%)')
ax.legend()
ax.grid(True, alpha=0.3)

# Отдельный график для МГУА
ax = axes[1, 1]
ax.plot(range(len(train_data), len(time_series)), test_data, color='black', label='Факт', linewidth=2)
ax.plot(range(len(train_data), len(time_series)), forecast_combi, color='orange', label='МГУА COMBI', linewidth=2)
ax.plot(range(len(train_data), len(time_series)), forecast_mia, color='purple', label='МГУА MIA', linewidth=2)
ax.set_title('Методы 3-4: МГУА (COMBI и MIA)', fontsize=12, fontweight='bold')
ax.set_xlabel('Время (индекс)')
ax.set_ylabel('Алкоголь (%)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('ПРОГНОЗИРОВАНИЕ ВРЕМЕННОГО РЯДА - СРАВНЕНИЕ МЕТОДОВ', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Функция для вычисления метрик
def evaluate_forecast(actual, predicted, name):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    print(f"\n📊 {name}:")
    print(f"   MSE:  {mse:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE:  {mae:.4f}")
    print(f"   R²:   {r2:.4f}")
    print(f"   MAPE: {mape:.2f}%")
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2, 'MAPE': mape}

# Оценка каждого метода
results = {}
results['ARIMA'] = evaluate_forecast(test_data, forecast_arima, "ARIMA")
results['Symbolic'] = evaluate_forecast(test_data, forecast_poly, "Символьная регрессия")
results['GMDH_COMBI'] = evaluate_forecast(test_data, forecast_combi, "МГУА COMBI")
results['GMDH_MIA'] = evaluate_forecast(test_data, forecast_mia, "МГУА MIA")

print("\n" + "=" * 80)
print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ МЕТОДОВ")
print("=" * 80)

# Создаем DataFrame для сравнения
comparison_df = pd.DataFrame({
    'Метод': ['ARIMA', 'Символьная регрессия', 'МГУА COMBI', 'МГУА MIA'],
    'RMSE': [results['ARIMA']['RMSE'], results['Symbolic']['RMSE'], 
             results['GMDH_COMBI']['RMSE'], results['GMDH_MIA']['RMSE']],
    'MAE': [results['ARIMA']['MAE'], results['Symbolic']['MAE'], 
            results['GMDH_COMBI']['MAE'], results['GMDH_MIA']['MAE']],
    'R²': [results['ARIMA']['R²'], results['Symbolic']['R²'], 
           results['GMDH_COMBI']['R²'], results['GMDH_MIA']['R²']],
    'MAPE (%)': [results['ARIMA']['MAPE'], results['Symbolic']['MAPE'], 
                 results['GMDH_COMBI']['MAPE'], results['GMDH_MIA']['MAPE']]
})

print("\n📊 Сравнительная таблица:")
print(comparison_df.to_string(index=False))

# Находим лучший метод
best_method = comparison_df.loc[comparison_df['RMSE'].idxmin(), 'Метод']
print(f"\n🏆 Лучший метод по RMSE: {best_method}")

# Визуализация сравнения
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# График 1: RMSE
methods = comparison_df['Метод']
rmse_values = comparison_df['RMSE']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
bars = axes[0].bar(methods, rmse_values, color=colors, edgecolor='black', linewidth=2)
axes[0].set_title('Сравнение RMSE', fontsize=12, fontweight='bold')
axes[0].set_ylabel('RMSE')
axes[0].grid(True, alpha=0.3)
for bar, value in zip(bars, rmse_values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', fontsize=10, fontweight='bold')

# График 2: R²
r2_values = comparison_df['R²']
bars = axes[1].bar(methods, r2_values, color=colors, edgecolor='black', linewidth=2)
axes[1].set_title('Сравнение R²', fontsize=12, fontweight='bold')
axes[1].set_ylabel('R²')
axes[1].grid(True, alpha=0.3)
for bar, value in zip(bars, r2_values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', fontsize=10, fontweight='bold')

plt.suptitle('СРАВНЕНИЕ МЕТОДОВ ПРОГНОЗИРОВАНИЯ', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

best_rmse = comparison_df.loc[comparison_df['RMSE'].idxmin()]
best_r2 = comparison_df.loc[comparison_df['R²'].idxmax()]

print(f"\n✅ Лучший метод по RMSE: {best_rmse['Метод']} (RMSE = {best_rmse['RMSE']:.4f})")
print(f"✅ Лучший метод по R²: {best_r2['Метод']} (R² = {best_r2['R²']:.4f})")
print(f"\n💡 Рекомендация: Для прогнозирования временных рядов на данных о вине")
print(f"   рекомендуется использовать метод {best_rmse['Метод']}")