import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Spam Classifier", page_icon="📩")

# Load Dataset with Auto Column Detection
@st.cache_data
def load_data():
    df = pd.read_csv("mail_data.csv", encoding="latin-1")

    # Auto-detect label and message columns
    label_col = [col for col in df.columns if "label" in col.lower() or "category" in col.lower() or "v1" in col.lower()][0]
    message_col = [col for col in df.columns if "message" in col.lower() or "text" in col.lower() or "v2" in col.lower()][0]

    df = df[[label_col, message_col]]
    df.columns = ["Label", "Message"]
    df["Label"] = df["Label"].map({"ham": 0, "spam": 1})

    return df

df = load_data()

# Display column names and sample data for verification
# st.write("📋 Columns detected:", df.columns.tolist())
# st.dataframe(df.head())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df["Message"], df["Label"], test_size=0.2, random_state=42)

# Train Model
@st.cache_resource
def train_model():
    model = make_pipeline(TfidfVectorizer(stop_words="english"), MultinomialNB())
    model.fit(X_train, y_train)
    return model

model = train_model()

# UI
st.title("📩 Spam Message Classifier")
st.write("Check if a message is **Spam** or **Ham (Not Spam)** using a trained ML model.")

# Optional sample input
if st.checkbox("Try sample messages"):
    sample = st.selectbox("Choose a sample:", [
        "Win a free vacation to Bahamas! Click now!",
        "Are we still meeting at 6?",
        "URGENT! Your account has been blocked. Click here to reactivate.",
        "Happy birthday! Hope you have a great year ahead."
    ])
    user_input = sample
else:
    user_input = st.text_area("Enter your message:")

# Prediction
if st.button("Check Spam"):
    if user_input:
        prediction = model.predict([user_input])[0]
        if prediction == 1:
            st.error("🚫 Prediction: **SPAM**")
        else:
            st.success("✅ Prediction: **HAM**")
    else:
        st.warning("⚠️ Please enter or select a message.")

# Model Evaluation
st.subheader("📊 Model Performance")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"**Model Accuracy:** {accuracy:.2f}")

st.write("### 📄 Classification Report")
st.text(classification_report(y_test, y_pred))

st.write("### 🔢 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"], ax=ax)
st.pyplot(fig)
