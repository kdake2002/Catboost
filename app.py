import streamlit as st
import pandas as pd
import pickle

# Load the model
with open("catboost.pkl", "rb") as f:
    model = pickle.load(f)

# Label mapping
label_map = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

# App title
st.title("Iris Flower Classification App 🌸")
st.write("Enter the flower measurements to predict the Iris species.")

# Input fields
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

# Prediction
if st.button("Predict"):
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                              columns=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"])
    prediction = model.predict(input_data)
    flower_name = label_map.get(int(prediction[0]), "Unknown")
    st.success(f"The predicted Iris species is: **{flower_name}**")
