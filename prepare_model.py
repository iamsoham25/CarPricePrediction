import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# --- Load Data ---
try:
    df = pd.read_csv("car_data.csv")
except FileNotFoundError:
    print("Error: 'car_data.csv' not found. Please ensure the dataset is in the same directory.")
    exit() # Exit if the data file is missing

# --- Data Cleaning ---
df['year'] = df['year'].astype(int)
df['km_driven'] = df['km_driven'].astype(int)
df['fuel'] = df['fuel'].astype(str)
df['seller_type'] = df['seller_type'].astype(str)
df['transmission'] = df['transmission'].astype(str)
df['owner'] = df['owner'].astype(str)

# Drop 'name' column (too many unique values)
df = df.drop('name', axis=1)

# --- One-Hot Encoding for Categorical Variables ---
# Use drop_first=True to avoid multicollinearity
df_encoded = pd.get_dummies(df, drop_first=True)

# --- Features and Target ---
X = df_encoded.drop('selling_price', axis=1)
y = df_encoded['selling_price']

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Model ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Save Model and Column Names ---
with open("car_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("columns.pkl", "wb") as f:
    # Save the list of column names used for training
    pickle.dump(X.columns.tolist(), f)

print("✅ Model and column structure saved successfully.")