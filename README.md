# 🏦 Customer Churn Prediction for Financial Institutions

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-red.svg)](https://xgboost.readthedocs.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)](https://scikit-learn.org/)

A machine learning project that predicts customer churn in financial institutions using various classification algorithms and advanced techniques like SMOTE and ensemble methods.

## 📊 Project Overview

This project implements multiple machine learning models to predict customer churn in banking institutions. The best-performing model achieves a recall of 0.59 for churned customers, making it particularly valuable for institutions where customer retention is a priority.

### Key Features

- Multiple classification algorithms comparison
- Feature engineering for enhanced prediction
- Handling class imbalance using SMOTE
- Ensemble methods implementation
- Comprehensive model evaluation metrics

## 🛠️ Technologies Used

- Python 3.7+
- pandas
- scikit-learn
- XGBoost
- seaborn
- matplotlib
- imbalanced-learn

## 📈 Models Implemented

- XGBoost Classifier
- Random Forest
- Decision Tree
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Logistic Regression
- Voting Classifier (Ensemble)

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost seaborn matplotlib imbalanced-learn
```

### Dataset

The project uses a customer churn dataset with the following features:
- Demographics (Age, Geography, Gender)
- Banking relationships (Balance, Credit Score, Tenure)
- Product usage (NumOfProducts, HasCrCard, IsActiveMember)
- Financial metrics (EstimatedSalary)

### Running the Models

```python
# Clone the repository
git clone https://github.com/stonewerner/customer-churn-ML.git
cd customer-churn-ML

```

## 📊 Feature Engineering

Several derived features were created to improve model performance:
- Customer Lifetime Value (CLV)
- Age Groups (Young, MiddleAge, Senior, Elderly)
- Tenure-Age Ratio
- One-hot encoded categorical variables

## 💡 Key Findings

- The ensemble model with SMOTE achieved the best recall (0.59) for churned customers
- Most important features for prediction:
  - Balance
  - Age
  - EstimatedSalary
  - Geography
  - NumOfProducts

## 📝 Model Selection Rationale

The final model prioritizes recall over precision because:
- The cost of losing a customer (false negative) is typically higher than the cost of retention actions on non-churning customers (false positive)
- Higher recall ensures we identify more potential churners, allowing proactive retention measures

## 🤝 Contributing

Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss the proposed changes.


## 👥 Contact

For questions or collaboration opportunities, please reach out to Stone Werner, stonewerner.com

## 🔍 Future Improvements

- [ ] Deep learning models implementation
- [ ] API deployment for real-time predictions
- [ ] Feature selection optimization
- [ ] Hyperparameter tuning
- [ ] Cross-validation implementation
