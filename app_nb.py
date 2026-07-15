import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

st.set_page_config(page_title="Naive Bayes Spam Classifier", layout="centered")
st.title("📧 Naive Bayes Spam Classifier")
st.write("Train a Naive Bayes model and see Accuracy, Precision, Recall, F1-Score + Confusion Matrix")

# 1. UPLOAD DATA
uploaded_file = st.file_uploader("Upload your spam.csv file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin-1')

    # Clean columns - assuming spam.csv has 'v1' and 'v2'
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    st.subheader("Dataset Preview")
    st.write(df.head())

    # 2. SPLIT DATA
    X = df['message']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. VECTORIZE
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 4. TRAIN MODEL
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    # 5. SHOW RESULTS
    st.subheader("📊 Model Performance")

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.write(f"*Accuracy:* {acc:.2f}")
    st.write(f"*Precision:* {precision:.2f}")
    st.write(f"*Recall:* {recall:.2f}")
    st.write(f"*F1-Score:* {f1:.2f}")

    # 6. CONFUSION MATRIX
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NOT SPAM', 'SPAM'],
                yticklabels=['NOT SPAM', 'SPAM'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)

    # 7. PREDICT NEW MESSAGE
    st.subheader("Test Your Own Message")
    user_input = st.text_area("Enter a message:")
    if st.button("Predict"):
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)
        if prediction[0] == 1:
            st.error("🚨 This is SPAM")
        else:
            st.success("✅ This is NOT SPAM")
else:
    st.info("Please upload a spam.csv file to start")
