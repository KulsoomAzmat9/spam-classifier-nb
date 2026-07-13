import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="Spam Classifier NB", layout="centered")

st.title("Spam Email Classifier - Naive Bayes")
st.write("Dataset: 105 Emails | Model Accuracy: 1.00")

# Load model and vectorizer
try:
    model = pickle.load(open('nb_model.pkl', 'rb'))
    vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
except:
    st.error("Model files not found. Please add nb_model.pkl and tfidf_vectorizer.pkl")

user_input = st.text_input("Enter email text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text first")
    else:
        input_data = vectorizer.transform([user_input])
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0].max()

        if prediction == 1:
            st.error(f"RESULT: SPAM | Confidence: {prob:.2f}")
        else:
            st.success(f"RESULT: NOT SPAM | Confidence: {prob:.2f}")

st.write("---")
st.caption("Trained on 105 emails. SPAM: 1.00 | NOT SPAM: 1.00")
