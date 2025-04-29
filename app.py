import joblib
from transform_text import transform_text
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import streamlit as st
from PIL import Image
import base64

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Spam Detection App",
    page_icon="📨",
    layout="centered",
    initial_sidebar_state="auto",
)

# -------------------- Header Image --------------------
header_image = Image.open("header_visual.PNG")
resized_image = header_image.resize((300, 80))  # width x height in pixels
st.image(resized_image)

# -------------------- Custom Dark Theme CSS --------------------
custom_css = """
<style>
    .main {
        background-color: #0e1117;
        color: #f0f2f6;
    }
    .stTextArea textarea {
        background-color: #1e2128;
        color: #f0f2f6;
        border: 1px solid #3a3f4b;
        border-radius: 8px;
        font-size: 16px;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #e84343;
    }
    h1, h2, h3, h4, h5, h6, p {
        color: #f0f2f6;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# -------------------- App Title --------------------
st.markdown("<h1 style='text-align: center;'>📨 Spam Detection App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Identify whether an incoming message is spam or safe in real time.</p>", unsafe_allow_html=True)

# -------------------- Load Artifacts --------------------
vocab     = joblib.load('vocab.joblib')
NB_model  = joblib.load('naive_bayes_model.joblib')

# Rebuild vectorizer
texts     = pd.read_csv('Spam_emails_cleaned.csv')['Transformed_text'].fillna('').tolist()
vectorizer = TfidfVectorizer(vocabulary=vocab)
vectorizer.fit(texts)

# -------------------- Input Area --------------------
input_sms = st.text_area("Enter the message", height=150)

# -------------------- Prediction --------------------
if st.button('Predict'):
    if not input_sms.strip():
        st.warning("Please enter a message to classify.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input    = vectorizer.transform([transformed_sms])
        with st.spinner('Analyzing...'):
            result = NB_model.predict(vector_input)[0]

        if result == 1:
            st.error("🚩 This message is classified as SPAM.")
        else:
            st.success("✅ This message is Not Spam.")

# -------------------- Footer --------------------
st.markdown("<hr style='margin-top: 30px; margin-bottom: 10px;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 14px;'>Built by <b>Rayyan Mustafa</b> • Spam Detector © 2025</p>", unsafe_allow_html=True)
