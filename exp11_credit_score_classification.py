"""
Experiment 11: Credit Score Classification
Description: Write a program for the task of Credit Score Classification.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def generate_credit_data(n_samples=500):
    np.random.seed(42)
    annual_income = np.random.uniform(20000, 150000, n_samples)
    monthly_balance = np.random.uniform(500, 10000, n_samples)
    num_bank_accounts = np.random.randint(1, 10, n_samples)
    num_credit_cards = np.random.randint(1, 12, n_samples)
    interest_rate = np.random.uniform(3, 30, n_samples)
    num_loans = np.random.randint(0, 8, n_samples)
    delay_due_days = np.random.randint(0, 45, n_samples)
    outstanding_debt = np.random.uniform(100, 5000, n_samples)
    credit_history_age = np.random.uniform(1, 25, n_samples)

    # Heuristic score calculation
    score_metric = (
        (annual_income / 1000) * 0.4
        + monthly_balance * 0.05
        + credit_history_age * 5.0
        - delay_due_days * 3.5
        - outstanding_debt * 0.03
        - interest_rate * 2.0
        - num_loans * 4.0
    )

    # Classify into 3 categories: Poor (0), Standard (1), Good (2)
    q33, q66 = np.percentile(score_metric, [33, 66])
    credit_score = np.where(score_metric < q33, 0, np.where(score_metric < q66, 1, 2))

    df = pd.DataFrame({
        'AnnualIncome': annual_income,
        'MonthlyBalance': monthly_balance,
        'NumBankAccounts': num_bank_accounts,
        'NumCreditCards': num_credit_cards,
        'InterestRate': interest_rate,
        'NumLoans': num_loans,
        'DelayDays': delay_due_days,
        'OutstandingDebt': outstanding_debt,
        'CreditHistoryAge': credit_history_age,
        'CreditScore': credit_score
    })
    return df


def main():
    print("=" * 60)
    print("EXPERIMENT 11: CREDIT SCORE CLASSIFICATION")
    print("=" * 60)

    df = generate_credit_data(n_samples=600)
    print("Sample Credit Records:")
    print(df.head().round(2).to_string(index=False))
    print("-" * 60)

    feature_cols = [
        'AnnualIncome', 'MonthlyBalance', 'NumBankAccounts', 'NumCreditCards',
        'InterestRate', 'NumLoans', 'DelayDays', 'OutstandingDebt', 'CreditHistoryAge'
    ]
    X = df[feature_cols]
    y = df['CreditScore']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    y_pred = rf_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    score_labels = ['Poor', 'Standard', 'Good']
    print(f"Credit Score Classification Accuracy: {acc * 100:.2f}%\n")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=score_labels))

    # Test on a new applicant
    applicant = pd.DataFrame([{
        'AnnualIncome': 95000,
        'MonthlyBalance': 4200,
        'NumBankAccounts': 2,
        'NumCreditCards': 3,
        'InterestRate': 7.5,
        'NumLoans': 1,
        'DelayDays': 2,
        'OutstandingDebt': 650,
        'CreditHistoryAge': 12
    }])

    applicant_scaled = scaler.transform(applicant)
    pred_score = rf_model.predict(applicant_scaled)[0]
    pred_prob = rf_model.predict_proba(applicant_scaled)[0]

    print("-" * 60)
    print("Evaluation for New Applicant:")
    for col, val in applicant.iloc[0].items():
        print(f"  {col}: {val}")
    print(f"\nPredicted Credit Score: {score_labels[pred_score]}")
    for label, prob in zip(score_labels, pred_prob):
        print(f"  Probability ({label}): {prob * 100:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
