# Titanic Survival Prediction with Logistic Regression

This repository implements a complete machine learning pipeline to predict passenger survival on the **Titanic dataset** using **logistic regression**.  
It covers **data exploration, cleaning, feature engineering, visualization, feature selection, model training, and evaluation** with modern ML practices.

---

## 📊 Project Overview
- Dataset: [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)  
- Goal: Predict survival (`Survived`) based on passenger demographics, ticket, and fare information.  
- Approach: Logistic Regression with feature selection via **LassoCV** and **Recursive Feature Elimination (RFE)**.  

---

## 🚀 Features
- **Exploratory Data Analysis (EDA):**
  - Summary statistics and dataset structure (`df.describe()`, `df.info()`).
  - Correlation heatmap, scatter plots (Age vs Fare), bar plots (Survival by Gender).
  - Survival analysis by **family size** and **travel status (alone vs with family)**.
- **Data Cleaning & Preprocessing:**
  - Handling missing values (mean/median/mode imputation).
  - Removing high-missingness columns (Cabin).
  - Encoding categorical variables (`Sex`, `Embarked`).
  - Feature scaling using `StandardScaler`.
- **Feature Selection:**
  - LassoCV for L1-based feature selection.
  - Recursive Feature Elimination (RFE) with Logistic Regression.
  - Common selected features: `Pclass`, `Age`, `SibSp`, `Sex_male`, `Embarked_S`.
- **Model Training & Evaluation:**
  - Logistic Regression on selected features.
  - Training Accuracy: **95%**, Test Accuracy: **96.6%**.
  - Confusion Matrix, ROC Curve (AUC = **0.88**).
  - Evaluation Metrics: Accuracy, Precision, Recall, F1-score.
- **Extended Analysis:**
  - Multinomial Logistic Regression (OvR vs Multinomial) for fare-based 3-class survival chance.
  - Coefficient interpretation with odds ratios for feature importance.

---

## 📂 Repository Structure
