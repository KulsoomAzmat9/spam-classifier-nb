import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📧 Spam Email Classifier - Naive Bayes")

# 1. LOAD DATA
try:
    df = pd.read_csv("email_data.csv") 
    st.success("✅ Loaded email_data.csv successfully!")
except FileNotFoundError:
    st.error("❌ email_data.csv not found")
    st.stop()

# 2. CLEAN DATA - This fixes the KeyError
st.subheader("1. Dataset Preview")
df = df.dropna(subset=['text', 'label'])  # remove rows with empty text or label
df['text'] = df['text'].astype(str)       # make sure text is string
df['label'] = df['label'].astype(int)     # make sure label is 0 or 1

st.dataframe(df.head())
st.write(f"*Total rows after cleaning:* {len(df)}")

# 3. TRAIN MODEL
st.subheader("2. Training Naive Bayes Model...")
X = df['text']
y = df['label']

vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

# 4. SAVE MODEL
joblib.dump(model, 'spam_model_nb.joblib')
joblib.dump(vectorizer, 'vectorizer_nb.joblib')
st.success("✅ Model Trained and Saved!")

# 5. SHOW RESULTS
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("3. Model Performance")
st.metric("Accuracy", f"{acc*100:.2f}%")

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# 6. CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
st.pyplot(fig)

st.balloons()
