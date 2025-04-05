📩 Spam Message Classifier using Machine Learning
This project is a spam message classifier that uses TF-IDF vectorization and a Multinomial Naïve Bayes model. It also uses GridSearchCV to tune hyperparameters and improve accuracy. The classifier predicts whether a given SMS/text message is spam or ham (not spam).

🚀 Features
Preprocessing of real-world SMS data

TF-IDF vectorizer with unigrams and bigrams

Hyperparameter tuning using GridSearchCV

Model evaluation: accuracy, classification report, and confusion matrix

Real-time message prediction via console input

📁 Dataset
Dataset used: mail_data.csv
The dataset contains two columns:

Label: spam or ham

Message: The actual SMS/text message content

🛠️ Requirements
Install required packages:

bash
Copy
Edit
pip install pandas numpy matplotlib seaborn scikit-learn
▶️ How to Run
bash
Copy
Edit
python app.py
Then type a message when prompted to check if it's SPAM or HAM. Type exit to quit.

🧠 Model Details
Vectorizer: TfidfVectorizer (with stopwords removal and bigram support)

Classifier: MultinomialNB

Tuning: GridSearchCV (alpha values: 0.1, 0.5, 1.0)

Train/Test Split: 80/20 with stratification

📊 Example Output
plaintext
Copy
Edit
Model Accuracy: 0.98

Classification Report:
              precision    recall  f1-score   support

         ham       0.98      1.00      0.99       965
        spam       0.99      0.90      0.94       150

    accuracy                           0.98      1115
📧 Contact
For any queries, suggestions, or collaborations, feel free to reach out:

Email: prabhunandan016@gmail.com
