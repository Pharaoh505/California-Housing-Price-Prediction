import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

@st.cache
def load_model():
    california = fetch_california_housing()
    X = california.data
    y = california.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = load_model()

st.title("California Housing Price Prediction")

MedInc = st.number_input("Median Income (in $10,000s)", min_value=0.0, max_value=20.0, value=5.0)
HouseAge = st.number_input("House Age (in years)", min_value=1, max_value=100, value=20)
AveRooms = st.number_input("Average Rooms", min_value=1, max_value=20, value=6)
AveOccup = st.number_input("Average Occupants", min_value=1, max_value=20, value=3)
Latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=33.5)
Longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-118.5)
MedVal = st.number_input("Median House Value (in $100,000s)", min_value=1.0, max_value=5.0, value=2.0)
AveHouseValue = st.number_input("Average House Value (in $100,000s)", min_value=1.0, max_value=5.0, value=5.0)

input_data = np.array([[MedInc, HouseAge, AveRooms, AveOccup, Latitude, Longitude, MedVal, AveHouseValue]])

if "predictions" not in st.session_state:
    st.session_state.predictions = []

prediction = model.predict(input_data)
st.session_state.predictions.append(prediction[0])

st.write(f"The predicted house price is: ${prediction[0] * 100000:.2f}")

if st.session_state.predictions:
    avg_price = np.mean(st.session_state.predictions)
    st.markdown("---")
    st.subheader("📊 Prediction Summary")
    st.write(f"Total Predictions Made: {len(st.session_state.predictions)}")
    st.write(f"Average Predicted Price: ${avg_price * 100000:.2f}")

    low = sum(p < 2 for p in st.session_state.predictions)
    mid = sum(2 <= p < 5 for p in st.session_state.predictions)
    high = sum(p >= 5 for p in st.session_state.predictions)
    total = len(st.session_state.predictions)

    st.write(f"% < $200k: {100 * low / total:.2f}%")
    st.write(f"% between $200k–$500k: {100 * mid / total:.2f}%")
    st.write(f"% > $500k: {100 * high / total:.2f}%")

    fig, ax = plt.subplots()
    ax.plot(st.session_state.predictions, marker='o', linestyle='-')
    ax.set_title("Predicted Prices Over Time")
    ax.set_ylabel("Price ($100,000s)")
    ax.set_xlabel("Prediction #")
    st.pyplot(fig)
