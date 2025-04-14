# Fake News Detection using NLP
# Author: Seeram Shanmukha
# Description: Optimized and visualized script for classifying fake news using NLP.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import warnings
warnings.filterwarnings("ignore")

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load and sample data for speed
df_fake = pd.read_csv('/content/Fake.csv').sample(n=3000, random_state=1)
df_fake['label'] = 0

df_real = pd.read_csv('/content/True.csv').sample(n=3000, random_state=1)
df_real['label'] = 1

data = pd.concat([df_fake, df_real], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

# Visualization: Class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=data, palette='Set2')
plt.title('Class Distribution (0 = Fake, 1 = Real)')
plt.xlabel('News Type')
plt.ylabel('Count')
plt.show()

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

data['content'] = data['text'].apply(preprocess)

# TF-IDF
vectorizer = TfidfVectorizer(max_df=0.7)
X = vectorizer.fit_transform(data['content'])
y = data['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
lr = LogisticRegression(max_iter=300)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

pac = PassiveAggressiveClassifier(max_iter=300)
pac.fit(X_train, y_train)
pac_pred = pac.predict(X_test)
pac_acc = accuracy_score(y_test, pac_pred)

# Accuracy comparison
print(f"\n✅ Logistic Regression Accuracy: {lr_acc * 100:.2f}%")
print(f"✅ Passive Aggressive Classifier Accuracy: {pac_acc * 100:.2f}%")

# Bar chart of model accuracy
plt.figure(figsize=(6, 4))
sns.barplot(x=["Logistic Regression", "PAC"], y=[lr_acc, pac_acc], palette='coolwarm')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0.8, 1.0)
plt.show()

# Best model
best_model_name = "PAC" if pac_acc > lr_acc else "Logistic Regression"
y_pred = pac_pred if pac_acc > lr_acc else lr_pred

print(f"\n🔥 Best Model: {best_model_name}")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
