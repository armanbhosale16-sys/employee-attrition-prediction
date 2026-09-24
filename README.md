# Employee Attrition Prediction

## Project Overview

This project is an end-to-end Machine Learning web application designed to predict whether an employee will Stay or Leave an organization based on various workplace parameters and personal metrics. It covers data collection, preprocessing, model building, evaluation, model saving, and deployment using Streamlit.

## Dataset

The dataset contains information about employees, including:

* Age
* Monthly Income
* Years at Company
* Job Satisfaction
* OverTime
* Job Level
* Work-Life Balance
* Years Since Last Promotion
* Previous Work Experience

The target variable is:

* Attrition (Stay / Leave)

The dataset contains employee records used for binary classification modeling.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Joblib

## Machine Learning Model
- **Algorithm:** Logistic Regression
- **Preprocessing:** Categorical encoding for `OverTime` and feature scaling using `StandardScaler`.
- **Pipeline Structure:** Data scaling and model inference integration to prevent data leakage and ensure accurate predictions.

The dataset is divided into:

* 80% Training Data
* 20% Testing Data

## Model Evaluation

The classification model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* ROC-AUC Score

### Results

* Accuracy: 0.82
* Precision: 0.00
* Recall: 0.00
* F1 Score: 0.00
* Confusion Matrix: `[[33, 0], [7, 0]]`
* ROC-AUC: 0.61

## Model Saving

The trained model and preprocessing scaler are serialized using Joblib and saved as binary artifacts in the `model/` directory:
- Path: `model/attrition_model.pkl`
- Path: `model/scaler.pkl`

## Streamlit Application

An interactive web application built using Streamlit that provides:
- A clean 2-column layout interface for entering employee details.
- Real-time employee attrition prediction (Stay / Leave).
- Stay and Leave probability score calculations.
- Input summary viewer and interpretation section embedded directly within the app.

### Input Features

* Age (Years)
* Monthly Income ($)
* Years at Company
* Job Satisfaction (1-4)
* OverTime (Yes/No)
* Job Level (1-5)
* Work-Life Balance (1-4)
* Years Since Last Promotion
* Previous Work Experience (Years)

The application displays the predicted attrition outcome along with class probabilities.

## Project Structure

```text
employee_attrition_prediction/
├── dataset/
│   └── employee_data.csv
├── model/
│   ├── attrition_model.pkl
│   └── scaler.pkl
├── Screenshots/
│
├── .gitignore
├── README.md
├── app.py
├── requirements.txt
└── train_model.py
```

## How to Run the Project

### 1. Install required libraries

```bash
python -m pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_model.py
```

### 3. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will open in the browser.

## Conclusion

The project demonstrates an end-to-end Machine Learning workflow for predicting employee attrition, from dataset preprocessing and model training to model evaluation, model saving, and deployment using Streamlit.