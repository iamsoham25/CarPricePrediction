import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

# --- Load Model and Columns ---
try:
    with open("car_price_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("columns.pkl", "rb") as f:
        columns = pickle.load(f)
except FileNotFoundError:
    st.error("Model files (car_price_model.pkl or columns.pkl) not found. Please run prepare_model.py first.")
    st.stop() # Stop the app if model files are missing

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Car Price Predictor", layout="centered", page_icon="🚗")

st.title("🚗 Car Price Predictor")
st.markdown("Predict the **selling price** of a car based on its features.")

# --- User Inputs ---
st.header("Car Details")
year = st.slider("Year of Purchase", min_value=1990, max_value=2025, value=2015, step=1)
km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000, step=1000)
fuel = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
seller_type = st.selectbox("Seller Type", ['Individual', 'Dealer', 'Trustmark Dealer'])
transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
owner = st.selectbox("Owner Type", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])

# --- Prepare Input for Prediction ---
def prepare_input(year, km_driven, fuel, seller_type, transmission, owner, all_columns):
    input_data = {col: 0 for col in all_columns}
    input_data['year'] = year
    input_data['km_driven'] = km_driven

    # Set one-hot encoded values
    fuel_col = f'fuel_{fuel}'
    seller_col = f'seller_type_{seller_type}'
    trans_col = f'transmission_{transmission}'
    owner_col = f'owner_{owner}'

    for col in [fuel_col, seller_col, trans_col, owner_col]:
        if col in input_data:
            input_data[col] = 1
    
    return pd.DataFrame([input_data])

input_df = prepare_input(year, km_driven, fuel, seller_type, transmission, owner, columns)

# --- Predict Button ---
if st.button("Predict Selling Price", help="Click to get the estimated price"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Selling Price: ₹{int(prediction):,}")
    st.balloons() # Just for fun!


### Prediction Trend Graph


st.header("Prediction Trend: Price vs. Kilometers Driven")
st.markdown("See how the predicted selling price changes with varying **Kilometers Driven**, keeping other selected features constant.")

# Generate data for the plot
km_range = np.linspace(0, 200000, 100) # From 0 to 200,000 km
prices_for_plot = []

# Create a base input DataFrame (using the current user inputs)
base_input_data = prepare_input(year, 0, fuel, seller_type, transmission, owner, columns).iloc[0] # km_driven set to 0 initially

for km in km_range:
    temp_input_data = base_input_data.copy()
    temp_input_data['km_driven'] = km
    temp_df = pd.DataFrame([temp_input_data])
    
    # Ensure columns match the model's training columns
    # This step is crucial if the order of columns might differ, but 'prepare_input' should handle it
    temp_df = temp_df[columns] 
    
    prices_for_plot.append(model.predict(temp_df)[0])

# Create the plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(km_range, prices_for_plot, color='skyblue', linewidth=2)
ax.set_title("Predicted Selling Price vs. Kilometers Driven")
ax.set_xlabel("Kilometers Driven (km)")
ax.set_ylabel("Predicted Selling Price (₹)")
ax.grid(True, linestyle='--', alpha=0.6)
ax.ticklabel_format(style='plain', axis='y') # Prevent scientific notation on y-axis
ax.ticklabel_format(style='plain', axis='x') # Prevent scientific notation on x-axis

# Highlight the user's specific input on the graph
current_predicted_price = model.predict(input_df)[0] # Get the price for the user's exact km_driven
ax.axvline(km_driven, color='red', linestyle=':', label=f'Your Input ({km_driven} km)')
ax.plot(km_driven, current_predicted_price, 'ro', markersize=8, label=f'Your Predicted Price (₹{int(current_predicted_price):,})')
ax.legend()


st.pyplot(fig)

st.markdown("---")
