#You are a data scientist working on an email filtering system for a large email service provider. The goal is to automatically categorize incoming emails into "Spam" or "Ham" 
#(non-spam) categories. The company has decided to leverage Natural Language Processing (NLP) techniques and a Multinomial Naive Bayes classifier for this task. How would you 
#design and implement a spam and ham classification system using NLP and ML alogorithm

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

data_url = "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
df = pd.read_csv(data_url, encoding='latin-1')

df.columns = ['label', 'message','unnamed: 2', 'unnamed: 3', 'unnamed: 4']
df = df[['label', 'message']] 

df['label'] = df['label'].map({'spam': 1, 'ham': 0})

print(df.head())

X = df['message']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

def predict_email_category(email):
    email_tfidf = vectorizer.transform([email])
    prediction = model.predict(email_tfidf)
    return "Spam" if prediction[0] == 1 else "Ham"

print(predict_email_category("Congratulations! You've won a $1000 gift card. Click here to claim now."))
print(predict_email_category("Hey, are we still meeting for lunch tomorrow?"))
