import streamlit as st
import pickle
import json
import numpy as np

# Load model and columns
with open("model/bangalore_home_price_model.pickle", "rb") as f:
    model = pickle.load(f)

with open("model/columns.json", "r") as f:
    data_columns = json.load(f)["data_columns"]

# Extract locations
locations = [col.replace("location_", "").title() for col in data_columns if col.startswith("location_")]

# Streamlit UI
st.set_page_config(page_title="Bangalore Price Predictor", layout="centered")
st.title("🏡 Bangalore Home Price Predictor")

location = st.selectbox("📍 Select Location", sorted(locations))
sqft = st.number_input("📐 Total Square Feet", min_value=100, max_value=10000, value=1000, step=50)
bath = st.slider("🛁 Number of Bathrooms", 1, 10, 2)
bhk = st.slider("🛏️ Number of Bedrooms (BHK)", 1, 10, 2)

if st.button("🚀 Predict Price"):
    # Get location index
    loc_index = data_columns.index(f"location_{location.lower()}") if f"location_{location.lower()}" in data_columns else -1

    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    predicted_price = round(model.predict([x])[0], 2)
    st.success(f"Estimated Price: ₹ {predicted_price} Lakhs")
