import streamlit as st
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

@st.cache
def load_model():
    #Load and train the model only once
    california = fetch_california_housing()
    X = california.data
    y = california.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

#Load the trained model
model = load_model()

#Streamlit app layout
st.title("California Housing Price Prediction")

#Input fields for the user
MedInc = st.number_input("Median Income (in $10,000s)", min_value=0.0, max_value=20.0, value=5.0)
HouseAge = st.number_input("House Age (in years)", min_value=1, max_value=100, value=20)
AveRooms = st.number_input("Average Rooms", min_value=1, max_value=20, value=6)
AveOccup = st.number_input("Average Occupants", min_value=1, max_value=20, value=3)
Latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=33.5)
Longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-118.5)
MedVal = st.number_input("Median House Value (in $100,000s)", min_value=1.0, max_value=5.0, value=2.0)
AveHouseValue = st.number_input("Average House Value (in $100,000s)", min_value=1.0, max_value=5.0, value=5.0)

#Prepare input data for prediction
input_data = np.array([[MedInc, HouseAge, AveRooms, AveOccup, Latitude, Longitude, MedVal, AveHouseValue]])

#Make prediction using the trained model
prediction = model.predict(input_data)

#Display the predicted house price
st.write(f"The predicted house price is: ${prediction[0] * 100000:.2f}")
