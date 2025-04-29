# ─── app.py ───
import joblib
from transform_text import transform_text
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import streamlit as st

st.title("Spam Detection App")

# 1) Load your saved artifacts
vocab     = joblib.load('vocab.joblib')                  # original vocabulary dict
NB_model  = joblib.load('naive_bayes_model.joblib')

# 2) Rebuild exactly the same vectorizer (same token‐to‐index map), then fit to compute idf_
df        = pd.read_csv('Spam_emails_cleaned.csv')
texts     = df['Transformed_text'].fillna('').tolist()
vectorizer = TfidfVectorizer(vocabulary=vocab)
vectorizer.fit(texts)  # this only recomputes idf_ on your texts

# 3) On each predict, just transform & call the model
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input    = vectorizer.transform([transformed_sms])
    result          = NB_model.predict(vector_input)[0]
    st.success("🚩 SPAM" if result == 1 else "✅ Not Spam")
