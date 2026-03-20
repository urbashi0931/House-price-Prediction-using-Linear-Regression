import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Read the CSV file (use raw string r"" or forward slashes / for Windows paths)
df = pd.read_csv(r"C:\Housing.csv")

# Map furnishing status to numbers
furnishing_map = {
    'furnished': 1,
    'semi-furnished': 2,
    'unfurnished': 3
}
df['furnishingstatus'] = df['furnishingstatus'].map(furnishing_map)

# Separate features and target
X = df.drop('price', axis=1)
y = df['price']

# List of categorical columns (excluding furnishingstatus since we mapped it to numbers)
categorical_cols = ['mainroad', 'guestroom', 'basement', 
                    'hotwaterheating', 'airconditioning','prefarea']

# Apply one-hot encoding for categorical columns
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Ensure furnishingstatus is included in features
if 'furnishingstatus' not in X_encoded.columns:
    X_encoded['furnishingstatus'] = df['furnishingstatus']

# Optional: Check first 10 rows
print(X_encoded.head(10))

#Feature engineering: interactions
# Example: area * bedrooms, area * bathrooms
X_encoded['area_bedrooms'] = X_encoded['area'] * X_encoded['bedrooms']
X_encoded['area_bathrooms'] = X_encoded['area'] * X_encoded['bathrooms']

# Scale numeric features
numeric_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'furnishingstatus', 'area_bedrooms', 'area_bathrooms']
scaler = StandardScaler()
X_encoded[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict prices on the test set
y_pred = model.predict(X_test)
y_pred_ceiling = np.ceil(y_pred)


# Check the first 10 predictions vs actual prices
print("Predicted prices:", y_pred_ceiling[:10])
print("Actual prices:   ", y_test.values[:10])

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")



# Initialize model random forest
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)

# Train model
rf_model.fit(X_train, y_train)

# Predict
y_pred_rf = rf_model.predict(X_test)
y_pred_rf_ceiling = np.ceil(y_pred_rf)

# Evaluate
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest MSE: {mse_rf:.2f}")
print(f"Random Forest R²: {r2_rf:.2f}")