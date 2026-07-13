# spam-classifier-nb
## Description
This project detects whether an email is SPAM or NOT SPAM using Multinomial Naive Bayes with TF-IDF vectorization.

## Dataset
- *Total Emails*: 105
- *Features*: Email text converted using TF-IDF

## Model Used
- *Algorithm*: Multinomial Naive Bayes
- *Vectorizer*: TF-IDF
## Results
- *Training Accuracy*: 1.00 (100%)
## Test Example
*Input*: win free money  
*Output*: SPAM

## How to Run
1. Install requirements: pip install streamlit scikit-learn pandas
2. Run the app: streamlit run app.py

## Conclusion
Naive Bayes performed perfectly on this dataset with 100% accuracy.
