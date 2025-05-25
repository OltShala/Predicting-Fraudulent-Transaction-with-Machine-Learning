💳 Fraud Detection with Machine Learning
This project demonstrates how machine learning algorithms can be used to detect fraudulent credit card transactions using real-world data. It tackles the challenges of imbalanced datasets, feature scaling, and model evaluation through a complete end-to-end ML pipeline in Python.

📁 Dataset
Source: Kaggle - Credit Card Fraud Detection

Description: The dataset contains 284,807 transactions, where 492 are fraudulent (~0.17%). Features are numerical values obtained from PCA transformation, along with Time, Amount, and Class labels.

🧠 Technologies Used
Python
Pandas, NumPy – Data manipulation
Matplotlib, Seaborn – Data visualization
Scikit-learn – Machine learning models and metrics
SMOTE (imblearn) – Handling class imbalance

🔍 Project Workflow
Data Import & Exploration
Read and inspect the dataset
Visualize the distribution of fraudulent vs. non-fraudulent transactions
Data Preprocessing
Feature scaling (Time and Amount)
Dropped irrelevant features
Train-test split (80-20 ratio)
Handling Class Imbalance
Applied SMOTE (Synthetic Minority Oversampling Technique) to balance the training data
Model Building

Implemented two classification models:

Logistic Regression
Random Forest Classifier

Trained both on the balanced dataset
Model Evaluation
Confusion Matrix
Classification Report (Precision, Recall, F1-Score)
ROC Curve & AUC Score for performance comparison

Visualization

Heatmaps for confusion matrices
ROC Curve comparison for both models

📊 Results
Both models performed well after resampling.

Random Forest yielded a higher ROC AUC Score, making it the better model for this classification task.

The use of SMOTE significantly improved recall for the minority (fraudulent) class.

📌 Key Takeaways
Fraud detection requires handling severely imbalanced data with techniques like SMOTE.

Evaluation metrics like ROC AUC and recall are crucial for judging model effectiveness in such domains.

Ensemble models like Random Forest tend to perform better in high-stakes classification problems.


📬 Contact
For questions, collaborations, or feedback, feel free to reach out via LinkedIn or email me at [your-email@example.com].

⭐ Acknowledgments
Dataset: Kaggle - Credit Card Fraud Detection

Libraries: scikit-learn, imbalanced-learn, seaborn

