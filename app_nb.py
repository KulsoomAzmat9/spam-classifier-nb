import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("Project 1: Naive Bayes Spam Classifier")

@st.cache_data
def load_data():
    df = pd.read_csv("email_data.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()
st.write(f"Total emails: {len(df)}")
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)
acc = accuracy_score(y_test, model.predict(X_test_vec))

st.write(f"Accuracy: {acc:.2f}")
email = st.text_area("Enter email text")
if st.button("Predict"):
    vec = vectorizer.transform([email])
    pred = model.predict(vec)[0]
    st.success("Spam" if pred==1 else "Not Spam")
