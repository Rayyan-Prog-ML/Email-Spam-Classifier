import joblib
from transform_text import transform_text
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
NB_model=joblib.load('naive_bayes_model.joblib')
# vectorizer=joblib.load("vectorizer.joblib")
vectorizer = TfidfVectorizer()

df= pd.read_csv('Spam_emails_cleaned.csv')
X = df['Transformed_text']
X = vectorizer.fit_transform(X)

import streamlit as st

st.title("Spam Detection App")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess the input SMS
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize the transformed SMS using the loaded vectorizer
    vector_input = vectorizer.transform([transformed_sms])

    # 3. Make predictions using the loaded model
    result = NB_model.predict(vector_input)[0]

    # 4. Display the prediction
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
