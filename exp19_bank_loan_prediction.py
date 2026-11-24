"""
Experiment 19: Naive Bayes Classification for Bank Loan Prediction
Description: Implementation of Naive Bayes classification for Bank Loan prediction.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score


def generate_bank_loan_data(n_samples=600):
    np.random.seed(42)
    age = np.random.randint(22, 65, n_samples)
    experience = np.clip(age - np.random.randint(20, 26, n_samples), 0, 45)
    income = np.random.uniform(20, 200, n_samples)  # Annual Income in $000s
    family_size = np.random.choice([1, 2, 3, 4], n_samples)
    cc_avg = np.random.uniform(0.1, 8.0, n_samples)  # Credit card spending in $000s/month
    education = np.random.choice([1, 2, 3], n_samples, p=[0.4, 0.35, 0.25])  # 1: Undergrad, 2: Grad, 3: Advanced
    mortgage = np.random.choice([0, 50, 150, 300, 500], n_samples, p=[0.6, 0.15, 0.12, 0.08, 0.05])
    securities_account = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    cd_account = np.random.choice([0, 1], n_samples, p=[0.94, 0.06])
    online_banking = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    credit_card_user = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])

    # Loan acceptance probability: high income, higher education, high CC spending, and CD account drive loan uptake
    z = (
        -4.5
        + 0.035 * income
        + 0.30 * cc_avg
        + 0.6 * (education - 1)
        + 1.8 * cd_account
        + 0.001 * mortgage
        - 0.02 * age
    )
    prob = 1.0 / (1.0 + np.exp(-z))
    personal_loan = (prob > 0.5).astype(int)

    df = pd.DataFrame({
        'Age': age,
        'Experience': experience,
        'Income_k$': income.round(1),
        'Family': family_size,
        'CCAvg_k$': cc_avg.round(2),
        'Education': education,
        'Mortgage_k$': mortgage,
        'SecuritiesAccount': securities_account,
        'CDAccount': cd_account,
        'Online': online_banking,
        'CreditCard': credit_card_user,
        'PersonalLoan': personal_loan
    })
    return df


def main():
    print("=" * 60)
    print("EXPERIMENT 19: NAIVE BAYES FOR BANK LOAN PREDICTION")
    print("=" * 60)

    df = generate_bank_loan_data(600)
    print("Sample Bank Loan Customer Records:")
    print(df.head().to_string(index=False))
    print(f"\nLoan Acceptance Distribution:\n{df['PersonalLoan'].value_counts().to_string()}")
    print("-" * 60)

    feature_cols = [
        'Age', 'Experience', 'Income_k$', 'Family', 'CCAvg_k$',
        'Education', 'Mortgage_k$', 'SecuritiesAccount', 'CDAccount', 'Online', 'CreditCard'
    ]
    X = df[feature_cols]
    y = df['PersonalLoan']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Train Gaussian Naive Bayes Classifier
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    y_pred = gnb.predict(X_test)
    y_prob = gnb.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Naive Bayes Model Accuracy: {acc * 100:.2f}%")
    print(f"ROC AUC Score: {roc_auc:.4f}\n")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Loan Rejected (0)', 'Loan Approved (1)']))
    print("-" * 60)

    # Predict for a high-potential loan customer
    new_customer = pd.DataFrame([{
        'Age': 35,
        'Experience': 11,
        'Income_k$': 145.0,
        'Family': 3,
        'CCAvg_k$': 4.5,
        'Education': 3,
        'Mortgage_k$': 100,
        'SecuritiesAccount': 0,
        'CDAccount': 1,
        'Online': 1,
        'CreditCard': 1
    }])

    pred_decision = gnb.predict(new_customer)[0]
    pred_prob = gnb.predict_proba(new_customer)[0][1]

    print("Prediction for Loan Applicant Profile:")
    for k, v in new_customer.iloc[0].items():
        print(f"  {k}: {v}")
    print(f"\nLoan Approval Probability: {pred_prob * 100:.2f}%")
    print(f"Final Decision: {'Approved' if pred_decision == 1 else 'Rejected'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
