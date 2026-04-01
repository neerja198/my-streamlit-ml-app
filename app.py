import streamlit as st
import tensorflow as tf
import numpy as np

st.title("🚀 TensorFlow Model Deployment Demo")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")

model = load_model()

user_input = st.number_input("Enter a number", value=1.0)

if st.button("Predict"):
    prediction = model.predict(np.array([[user_input]]))
    st.success(f"Prediction: {prediction[0][0]:.2f}")
