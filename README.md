# Email Spam Classifier using Naive Bayes
A machine learning web application that classifies emails as Spam or Not Spam using the Naive Bayes algorithm. Built with Python, Scikit-learn.

## 1. Project Description
This project detects spam emails using text classification. The model is trained on an email dataset using TF-IDF Vectorizer and Multinomial Naive Bayes. The app provides a simple web interface where you can paste any email text and get an instant prediction.

## 2. Features
- Text Preprocessing: Removes punctuation, stopwords, and converts to lowercase
- TF-IDF Vectorization: Converts text to numerical features
- Naive Bayes Model: Fast and accurate for text classification
- Model Persistence: Uses joblib to save model_nb.joblib and vectorizer.joblib

## 3. Technologies Used
- *Language*: Python 
- *Libraries*: Scikit-learn, Pandas, NumPy, Joblib
- *Vectorizer*: TfidfVectorizer

## 4. Comparison of Algorithms

For this Spam Classifier project, I compared 3 algorithms on the email dataset:

### Naive Bayes - Final Model Used
- Type: Probabilistic classifier based on Bayes' Theorem
- Why used: Works extremely well for text classification with word frequency
- Pros: Very fast to train, handles high-dimensional text data, great for spam.
- 
- ### 2. Logistic Regression
- Type: Linear model for classification
- Why use: Good baseline for text classification
- Pros: Simple, interpretable, gives probability scores
- Cons: Can overfit with too many features. Needs regularization
  
### 3. Support Vector Machine (SVM)
- *Type*: Finds the best hyperplane to separate classes
- *Why used*: Very effective for high-dimensional text data
- *Pros*: Handles complex boundaries, good accuracy
- *Cons*: Slower to train on large datasets, harder to tune
  
### Conclusion
Naive Bayes was chosen for the final implementation because it provides a great balance of speed, simplicity for spam detection and accuracy of 100%, While SVM is much slower to train. Logistic Regression is a good alternative but Naive Bayes handles word frequency better for emails.
