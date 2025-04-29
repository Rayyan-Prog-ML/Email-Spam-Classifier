from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import nltk
import string

nltk.download('stopwords')
stopwords.words('english')
def transform_text(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))  # Set of stopwords
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # Step 1: Remov e URLs
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
