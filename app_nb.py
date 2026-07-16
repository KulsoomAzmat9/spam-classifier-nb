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
    st.error("❌ email_data.csv not found")
    st.stop()

# 2. FIX COLUMN NAMES + SHOW DATA
df.columns = df.columns.str.strip().str.lower()

st.subheader("1. Dataset Information")

st.write("Columns in file:")
st.code(list(df.columns)) # This will show the columns

st.write("Data Preview - First 5 Rows:")
st.dataframe(df.head())

# 3. CLEAN & TRAIN
df = df.dropna(subset=['text', 'label'])
df['text'] = df['text'].astype(str)
df['label'] = df['label'].astype(int)

st.subheader("2. Training Model...")
st.write(f"*Total rows used for training:* {len(df)}")

X = df['text']
y = df['label']
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

joblib.dump(model, 'spam_model_nb.joblib')
joblib.dump(vectorizer, 'vectorizer_nb.joblib')
st.success("✅ Model Trained and Saved!")

# 4. SHOW RESULTS
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("3. Model Performance")
st.metric("Accuracy", f"{acc*100:.2f}%")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))
st.write("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
st.pyplot(fig)

# 5. PREDICTION SECTION
st.subheader("4. Test  Email")
user_input = st.text_area("✍️ Type an email here:")

if st.button("🔍 Predict"):
    if user_input:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]

        if prediction == 1:
            st.error("🚨 This is SPAM!")
        else:
            st.success("✅ This is NOT Spam")
    else:
        st.warning("Please enter some text first")
