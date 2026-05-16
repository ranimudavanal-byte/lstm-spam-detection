import streamlit as st
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.title("Spam Detection App (LSTM)")

# Load model
model = load_model("spam_model.keras")

# Load tokenizer
tokenizer = joblib.load("tokenizer.joblib")

# Input
message = st.text_area("Enter your message")

# Predict button
if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message")
    else:
        # Convert text → numbers
        seq = tokenizer.texts_to_sequences([message])
        padded = pad_sequences(seq, maxlen=100)

        # Prediction
        prediction = model.predict(padded)

        # Output
        if prediction[0][0] > 0.5:
            st.error("Spam Message")
        else:
            st.success("Not Spam")