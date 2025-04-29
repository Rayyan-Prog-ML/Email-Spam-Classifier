import joblib
from transform_text import transform_text
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import streamlit as st

# Load model
NB_model = joblib.load('naive_bayes_model.joblib')
df = pd.read_csv('Spam_emails_cleaned.csv')
texts = df['Transformed_text'].fillna('').tolist()
vectorizer = TfidfVectorizer()
vectorizer.fit(texts)

# Streamlit app
st.title("Spam Detection App")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess the input SMS
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize using the loaded, pre-fitted vectorizer
    vector_input = vectorizer.transform([transformed_sms])

    # 3. Make prediction
    result = NB_model.predict(vector_input)[0]

    # 4. Display the result
    if result == 1:
        st.header("Spam 🚩")
    else:
        st.header("Not Spam ✅")
