import streamlit as st
import re
import pickle
import numpy as np
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk
import tensorflow as tf

# Ensure nltk resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load saved models
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Load ANN model (assumed to be saved in HDF5 format)
ann_model = tf.keras.models.load_model('ann_model.h5')

# Initialize NLP tools
tokenizer = TweetTokenizer()
stemmer = SnowballStemmer('english')
stop_words = set(stopwords.words('english'))

def preprocess_text(input_text):
    # Tokenize, clean, and stem
    processed_text = ' '.join(tokenizer.tokenize(input_text))
    processed_text = re.sub('[^a-zA-Z0-9]+', ' ', processed_text)
    processed_text = ' '.join([w for w in word_tokenize(processed_text) if len(w) >= 3])
    processed_text = ' '.join([stemmer.stem(w.lower()) for w in tokenizer.tokenize(processed_text)])
    processed_text = ' '.join([w for w in word_tokenize(processed_text) if w not in stop_words])
    return processed_text

def predict_bullying(text):
    processed = preprocess_text(text)
    vectorized = tfidf_vectorizer.transform([processed])
    prediction = ann_model.predict(vectorized)[0][0]
    return "Bullying 😠" if prediction > 0.5 else "Not Bullying 🙂"

# Streamlit UI
st.title("Cyberbullying Detection App")
st.write("Enter any message or comment to check if it's considered cyberbullying.")

user_input = st.text_area("Enter text here:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        result = predict_bullying(user_input)
        st.success(f"The message is classified as: **{result}**")
