📩 Spam Message Classifier
This project is a Spam Message Classifier that uses Machine Learning techniques to predict whether a given SMS or text message is SPAM or HAM (not spam). It includes a Streamlit web application for real-time message prediction and also supports command-line usage.

🔍 Overview
✅ Text preprocessing of real-world SMS data

✳️ TF-IDF vectorization with unigrams and bigrams

🤖 Multinomial Naive Bayes classifier

🔧 Hyperparameter tuning using GridSearchCV

📈 Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix

🌐 Web app using Streamlit for easy use

📂 Dataset
The dataset used is mail_data.csv and contains the following columns:

Label: The message type (either ham or spam)

Message: The text content of the message

🧠 Model Details
Vectorizer: TfidfVectorizer

Stopword removal

N-gram range: (1, 2)

Classifier: MultinomialNB

Hyperparameter Tuning: GridSearchCV

Alpha values: [0.1, 0.5, 1.0]

Train/Test Split: 80% training, 20% testing with stratification

📊 Performance
Evaluation Metrics:
Accuracy : 0.99
Precision: 0.99
Recall   : 0.91
F1-score : 0.95

Detailed Classification Report:
              precision    recall  f1-score   support

         Ham       0.99      1.00      0.99       966
        Spam       0.99      0.91      0.95       149

    accuracy                           0.99      1115
   macro avg       0.99      0.96      0.97      1115
weighted avg       0.99      0.99      0.99      1115


🚀 Getting Started
Clone the Repository
Start by cloning the repository to your local machine.

Install Dependencies
To run the project, you need to install the required libraries. These can be installed via the following command:

Install Python dependencies:
Install all the required dependencies manually by using a package manager like pip. You will need the following packages:

pandas

numpy

matplotlib

seaborn

scikit-learn

streamlit

Run the App
Web Interface (Recommended)
To start the Streamlit app and interact with the spam message classifier, you can run the application via the Streamlit interface.

Console Mode
You can also interact with the model using the command-line interface by running the provided Python script.

📎 Project Structure
bash
Copy
Edit
├── app.py                # Streamlit app
├── spam_classifier.py    # Console-based interface
├── mail_data.csv         # Dataset
├── README.md             # Project documentation

💬 Contact
Developer: Prabhunandan
📧 Email: prabhunandan016@gmail.com

Feel free to reach out for feedback, suggestions, or collaboration!
