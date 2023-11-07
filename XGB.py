import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd

# Загрузите данные в Pandas DataFrame
data = pd.read_csv("dataset.csv")

# Разделите данные на признаки и целевую переменную
X = data.drop(columns=["Мужчины в возрасте 0-15 лет"])
y = data["2"]

# Разделите данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создайте и обучите модель XGBoost
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# Сделайте прогнозы на тестовом наборе данных
y_pred = model.predict(X_test)

# Вычислите RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("RMSE:", rmse)
