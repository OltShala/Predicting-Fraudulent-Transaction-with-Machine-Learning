# 📦 Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from imblearn.over_sampling import SMOTE

# 📥 Load Dataset
df = pd.read_csv('creditcard.csv')

# 📊 Initial Exploration
print("Dataset shape:", df.shape)
print(df['Class'].value_counts())

# 📈 Class Distribution Plot
sns.countplot(x='Class', data=df)
plt.title("Fraud vs Non-Fraud Class Distribution")
plt.show()

# 🧼 Feature Engineering &  Standardization
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df[['Amount']])
df['scaled_time'] = scaler.fit_transform(df[['Time']])
df.drop(['Time', 'Amount'], axis=1, inplace=True)

# 📑 Prepare Features & Labels
X = df.drop('Class', axis=1)
y = df['Class']

# 🧪 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 🔁 Handle Imbalance with SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

print("After resampling:", pd.Series(y_res).value_counts())

# 🔁 Initialize Models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

# 📊 Evaluation Function
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n🧠 {name} Classification Report:\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    return fpr, tpr, auc

# 🧠 Train and Evaluate Both Models
roc_results = {}

for name, model in models.items():
    model.fit(X_res, y_res)
    fpr, tpr, auc = evaluate_model(name, model, X_test, y_test)
    roc_results[name] = (fpr, tpr, auc)

# 📈 Compare ROC Curves
plt.figure(figsize=(8, 6))
for name, (fpr, tpr, auc) in roc_results.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid()
plt.show()
