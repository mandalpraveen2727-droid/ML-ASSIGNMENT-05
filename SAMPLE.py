from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score)
# 1. Load Data
data = load_breast_cancer()
X = data.data
y = data.target

# 2. Train-Test Split (Stratified)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
# 3. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# 4. Model Training
model = LogisticRegression(max_iter=1000,random_state=42)
model.fit(X_train, y_train)

# 5. Prediction
y_pred = model.predict(X_test)

# 6. Evaluation Metrics
print("--- Logistic Regression Performance ---")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : "f"{precision_score(y_test, y_pred):.4f}")
print(f"Recall    : "f"{recall_score(y_test, y_pred):.4f}")
print(f"F1 Score  : "f"{f1_score(y_test, y_pred):.4f}")