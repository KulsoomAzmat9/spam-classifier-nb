import streamlit as st
import pandas as pd
import joblib
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
    st.error("❌ email_data.csv not found. Please upload it to GitHub.")
    st.stop()

# 2. SHOW DATA
st.subheader("1. Dataset Preview")
st.write("*Columns:*", list(df.columns))
st.dataframe(df.head())

# 3. CLEAN DATA
df = df.dropna()  # remove any row with empty cell
df['text'] = df['text'].astype(str)
df['label'] = df['label'].astype(int)

st.subheader("2. Training Model...")
st.write(f"*Total rows used:* {len(df)}")

# 4. TRAIN
X = df['text']
y = df['label']
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

joblib.dump(model, 'spam_model_nb.joblib')
joblib.dump(vectorizer, 'vectorizer_nb.joblib')
st.success("✅ Model Trained!")

# 5. RESULTS
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("3. Results")
st.metric("Accuracy", f"{acc*100:.2f}%")
st.text(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
st.pyplot(fig)
