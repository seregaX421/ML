import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = pd.DataFrame({
    'age': [25, 32, 41, 29, np.nan, 38, 45, 27, 33, 52],
    'salary': [50000, 85000, 120000, 60000, np.nan, 95000, 150000, 55000, 75000, 180000],
    'experience': [2, 8, 15, 4, np.nan, 10, 20, 3, 7, 25],
    'education': ['Bachelor', 'Master', 'PhD', 'Bachelor', np.nan, 'Master', 'PhD', 'Bachelor', 'Master', 'PhD'],
    'position': ['Junior', 'Middle', 'Senior', 'Junior', np.nan, 'Middle', 'Senior', 'Junior', 'Middle', 'Senior'],
    'department': ['IT', 'HR', 'IT', 'Sales', 'IT', 'Marketing', 'IT', 'Sales', 'HR', 'IT'],
    'city': ['Moscow', 'SPb', 'Moscow', 'Kazan', 'Moscow', 'SPb', 'Novosibirsk', 'Kazan', 'Moscow', 'SPb']
})

print("Исходные данные:")
print(data)
print("\n" + "="*60)

# ОБРАБОТКА ПРОПУСКОВ
print("Пропуски до обработки:")
print(data.isnull().sum())

numeric_cols = ['age', 'salary', 'experience']
for col in numeric_cols:
    data[col] = data[col].fillna(data[col].median())

categorical_cols = ['education', 'position', 'department', 'city']
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

print("\nПропуски после обработки:")
print(data.isnull().sum())
print("\nДанные после обработки пропусков:")
print(data)

# КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ
label_encoder = LabelEncoder()
data['education_encoded'] = label_encoder.fit_transform(data['education'])
data['position_encoded'] = label_encoder.fit_transform(data['position'])

data = pd.get_dummies(data, columns=['department', 'city'], prefix=['dept', 'city'])

data = data.drop(['education', 'position'], axis=1)

print("Данные после кодирования категориальных признаков:")
print(data)
print("\nТипы данных после кодирования:")
print(data.dtypes)

# МАСШТАБИРОВАНИЕ ДАННЫХ
features_to_scale = ['age', 'salary', 'experience', 'education_encoded', 'position_encoded']

one_hot_cols = [col for col in data.columns if col.startswith(('dept_', 'city_'))]
features_to_scale.extend(one_hot_cols)

print("Признаки для масштабирования:")
print(features_to_scale)

scaler = StandardScaler()
data_scaled = data.copy()
data_scaled[features_to_scale] = scaler.fit_transform(data_scaled[features_to_scale])

print("\nРезультат масштабирования (первые 5 строк):")
print(data_scaled.head())

print("\nСтатистика после масштабирования:")
print("Средние значения:")
print(data_scaled[features_to_scale].mean().round(10))
print("\nСтандартные отклонения:")
print(data_scaled[features_to_scale].std(ddof=0).round(6))