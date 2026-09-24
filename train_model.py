import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# Directory Creation
os.makedirs('dataset', exist_ok=True)
os.makedirs('model', exist_ok=True)

# 1. Data Collection / Load Dataset from 'dataset' folder
csv_path = os.path.join('dataset', 'employee_data.csv')
try:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise pd.errors.EmptyDataError
except (FileNotFoundError, pd.errors.EmptyDataError):
    print("Dataset not found or empty. Generating sample dataset...")
    np.random.seed(42)
    data = {
        'Age': np.random.randint(22, 60, 200),
        'MonthlyIncome': np.random.randint(2000, 20000, 200),
        'YearsAtCompany': np.random.randint(1, 15, 200),
        'JobSatisfaction': np.random.randint(1, 5, 200),
        'OverTime': np.random.choice(['Yes', 'No'], 200),
        'JobLevel': np.random.randint(1, 5, 200),
        'WorkLifeBalance': np.random.randint(1, 5, 200),
        'YearsSinceLastPromotion': np.random.randint(0, 10, 200),
        'PreviousWorkExperience': np.random.randint(0, 15, 200),
        'Attrition': np.random.choice(['Yes', 'No'], 200, p=[0.2, 0.8])
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

# 2. Data Preprocessing
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

le_overtime = LabelEncoder()
df['OverTime'] = le_overtime.fit_transform(df['OverTime'])
df['Target'] = df['Attrition'].apply(lambda x: 1 if x in ['Yes', 'Leave'] else 0)

X = df[['Age', 'MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction', 
        'OverTime', 'JobLevel', 'WorkLifeBalance', 
        'YearsSinceLastPromotion', 'PreviousWorkExperience']]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Building
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 4. Model Evaluation
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("--- Model Evaluation Metrics ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.2f}")
print(f"Recall: {recall_score(y_test, y_pred, zero_division=0):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred, zero_division=0):.2f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.2f}")

# 5. Save Model into 'model' folder
joblib.dump(model, os.path.join('model', 'attrition_model.pkl'))
joblib.dump(scaler, os.path.join('model', 'scaler.pkl'))
print("\nModel and Scaler successfully saved in 'model/' directory!")