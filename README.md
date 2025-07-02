# Fraud Detection Using Machine Learning: A Case Study with Python, MLflow, and Tableau

## Overview

This project demonstrates a full end-to-end application of machine learning for detecting fraudulent financial transactions in a large-scale dataset. It combines advanced data engineering, supervised learning, experiment tracking, and interactive visualization.

### Problem

Fraudulent transactions represent a small but critical threat in digital payment systems. Detecting these rare events requires careful feature engineering, robust modeling, and continuous monitoring.

### Dataset

- **Size:** 6.3 million transaction records
- **Fields:** Transaction type, amount, sender/receiver balances, and fraud flags
- **Source:** Synthetic dataset simulating real-world mobile money transfers

### Approach

- **Feature Engineering:** Derived multiple behavioral and transactional features (e.g., balance inconsistencies, user transaction history, and self-transfer flags)
- **Modeling:** Trained an XGBoost classifier optimized for highly imbalanced data using Python and scikit-learn
- **Experiment Tracking:** Used MLflow to track model parameters, metrics, and artifacts
- **Visualization:** Built an interactive Tableau dashboard to explore fraud patterns and support model findings

### Key Highlights

- Achieved high recall on fraud detection to minimize false negatives
- Identified fraud concentration in specific transaction types (TRANSFER and CASH_OUT)
- Developed a real-time feature generation function for deployment readiness
- Saved trained model and encoders for reproducible prediction

### Tools Used

- Python (pandas, scikit-learn, xgboost, MLflow)
- Tableau for interactive analysis
- Git & GitHub for version control
