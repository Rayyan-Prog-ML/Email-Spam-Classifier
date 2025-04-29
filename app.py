import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords.words('english')
import string
string.punctuation

# Data loading

file_path="Spam_emails.csv"
df= pd.read_csv(file_path)
df.info()
df.shape
print(df.sample(5))

# Data cleaning

# removing unused column
df.drop(columns=["Unnamed: 0"], inplace=True)
df.info()
df.rename(columns={'Body':'Text'}, inplace=True)
df.sample(5)
df['Text'] = df['Text'].astype(str)  # Ensure all values are strings

#checking for missing values and dropping it
df.isnull().sum()
df=df.dropna()
df = df[df['Text'].str.strip().astype(bool)]  # Keep rows where 'Text' is not empty

#REMOVE URLS FROM THE TEXT COLUMN
#Data preprocessing
import nltk
from nltk.corpus import stopwords
import string
#for stemming to bring the root word
from nltk.stem.porter import PorterStemmer
stemmer=PorterStemmer()
def transform_text(text):
    stop_words = set(stopwords.words('english'))  # Set of stopwords
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # Step 1: Remove URLs
    text = re.sub(url_pattern, " ", text)

    # Step 2: Remove special characters
    random_pattern = r'[^a-zA-Z0-9\s]'
    text = re.sub(random_pattern, " ", text)

    # Step 3: Convert to lowercase and remove extra spaces
    text = text.lower().strip()
    text = re.sub(r'\s+', " ", text)

    # Step 4: Remove stopwords and punctuation  implementing stem
    words = text.split()
    filtered_words = [stemmer.stem(word) for word in words if word not in stop_words and word not in string.punctuation]
    return " ".join(filtered_words)

df['Transformed_text']=df['Text'].apply(transform_text)
df.head()

#wordcloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud
wc=WordCloud(width=500, height=500, min_font_size=10, background_color='white')
spam_wc=wc.generate(df[df['Label']==1]['Transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(15,16))
plt.imshow(spam_wc)

ham_wc=wc.generate(df[df['Label']==0]['Transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(15,16))
plt.imshow(ham_wc)

# retreiving top features
spam_mssg=df[df['Label']==1]['Transformed_text']

vectorizer=TfidfVectorizer(max_features=30, stop_words='english')
x=vectorizer.fit_transform(spam_mssg)
spam_top_words=vectorizer.get_feature_names_out()
print(f"top spam words: {spam_top_words}")

# retreiving top features
ham_mssg=df[df['Label']==0]['Transformed_text']

vectorizer=TfidfVectorizer(max_features=30, stop_words='english')
x=vectorizer.fit_transform(ham_mssg)
ham_top_words=vectorizer.get_feature_names_out()
print(f"top ham words: {ham_top_words}")

# check for duplicated values
df.duplicated().sum()
df=df.drop_duplicates(keep='first')
df.shape
df.info()
df.sample(10)
#EDA
df['Label'].value_counts()
df.head()

#visualizing the data
import matplotlib.pyplot as plt
plt.pie(df['Label'].value_counts(), labels=["ham","spam"],autopct='%0.2f' )
plt.show()

# applying smote and random forest

from imblearn.over_sampling import SMOTE

# Separate features (X) and target variable (y)
X = df['Transformed_text']  # Assuming 'Text' column contains your text data
y = df['Label']

# Apply TF-IDF vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Apply SMOTE to oversample the minority class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Now X_resampled and y_resampled contain the balanced dataset
# You can proceed with model training using these resampled data

# Example: Splitting the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Example: Train a RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
rf_y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, rf_y_pred))

# Naive beyes
NB_model = MultinomialNB()
NB_model.fit(X_train, y_train)
nb_y_test_pred = NB_model.predict(X_test)
nb_y_train_pred = NB_model.predict(X_train)

# Evaluate the model
print(classification_report(y_test, nb_y_test_pred))

# Confusion Matrix

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, nb_y_test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Ham', 'Predicted Spam'],
            yticklabels=['Actual Ham', 'Actual Spam'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Calculate the training accuracy
train_accuracy = accuracy_score(y_train, nb_y_train_pred)
print(f"Training Accuracy: {train_accuracy}")

# Calculate the testing accuracy
test_accuracy = accuracy_score(y_test, nb_y_test_pred)
print(f"Testing Accuracy: {test_accuracy}")

# Checking if the model if overfitting

if train_accuracy > test_accuracy + 0.1:
    print("The model is overfitting.")
else:
    print("The model is not overfitting.")

import joblib

joblib.dump(NB_model, 'naive_bayes_model.joblib')

joblib.dump(vectorizer, 'vectorizer.joblib')

NB_model=joblib.load('naive_bayes_model.joblib')
vectorizer=joblib.load('vectorizer.joblib')

import streamlit as st
# Streamlit app
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

