# train_model.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load data
df = pd.read_csv("data/car_data.csv")

# Feature engineering
df['Car_Age'] = 2025 - df['Year of manufacture']
features = ['Manufacturer', 'Model', 'Fuel type', 'Engine size', 'Mileage', 'Car_Age']
target = 'Price'

# Encode categorical
encoders = {}
for col in ['Manufacturer', 'Model', 'Fuel type']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Scale numerical
scaler = StandardScaler()
df[['Engine size', 'Mileage', 'Car_Age']] = scaler.fit_transform(df[['Engine size', 'Mileage', 'Car_Age']])

# Train model
X = df[features]
y = df[target]
model = RandomForestRegressor()
model.fit(X, y)

# Save
joblib.dump(model, "models/price_model.pkl", compress=3)
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(encoders, 'models/encoders.pkl')