# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Load Dataset & Detect Column Names
df = pd.read_csv("mail_data.csv", encoding="latin-1")

# Print available columns to detect correct names
print("Available Columns in CSV:", df.columns)

# Auto-detect column names for labels and messages
for col in df.columns:
    if "label" in col.lower() or "category" in col.lower() or "v1" in col.lower():
        label_col = col
    if "message" in col.lower() or "text" in col.lower() or "v2" in col.lower():
        message_col = col

# Rename columns for consistency
df = df[[label_col, message_col]]
df.columns = ["Label", "Message"]

# Convert labels to binary (spam = 1, ham = 0)
df["Label"] = df["Label"].map({"ham": 0, "spam": 1})

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(df["Message"], df["Label"], test_size=0.2, random_state=42, stratify=df["Label"])

# Define a Pipeline with feature extraction and model training
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
    ("nb", MultinomialNB())
])

# Define hyperparameters to tune
param_grid = {
    "nb__alpha": [0.1, 0.5, 1.0],
}

# Use GridSearchCV to find the best model
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model after tuning
best_model = grid_search.best_estimator_

# Make Predictions
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]  # Probability of being spam

# Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nEvaluation Metrics:")
print(f"Accuracy : {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall   : {recall:.2f}")
print(f"F1-score : {f1:.2f}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))


# Real-time Spam Prediction Function with Threshold
def predict_spam(message, threshold=0.5):
    prob = best_model.predict_proba([message])[0][1]  # Probability of being spam
    return "SPAM" if prob >= threshold else "HAM"

# Interactive Console Input for Testing
while True:
    msg = input("\nEnter a message to check (or type 'exit' to quit): ")
    if msg.lower() == "exit":
        break
    result = predict_spam(msg, threshold=0.5)  # You can adjust threshold here
    print(f"Prediction: {result}")
    
    
# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()


