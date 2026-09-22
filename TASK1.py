import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

data = {
    "Study_Hours": [
        1.0, 1.5, 2.0, 2.5, 3.0,
        3.5, 4.0, 4.5, 5.0, 5.5,
        6.0, 6.5, 7.0, 7.5, 8.0,
        8.5, 9.0, 2.0, 4.0, 6.0,
        3.0, 5.0, 7.0, 8.0, 1.5,
        4.5, 5.5, 6.5, 7.5, 9.5
    ],

    "Attendance": [
        45, 50, 55, 58, 60,
        62, 65, 68, 72, 75,
        78, 80, 82, 85, 88,
        90, 92, 48, 55, 65,
        52, 68, 75, 80, 40,
        60, 70, 73, 83, 95
    ],

        "Result": [
        0, 0, 0, 0, 0,
        0, 0, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 0, 0, 1,
        0, 1, 1, 1, 0,
        0, 1, 1, 1, 1
    ]
}

df = pd.DataFrame(data)

print("Student Dataset:")
print(df)

print("\nResult Distribution:")
print(df["Result"].value_counts())


X = df[["Study_Hours", "Attendance"]]
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=42,stratify=y)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled,y_train)

y_pred = model.predict(X_test_scaled)
y_probability = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("\nLogistic Regression Performance")
print("-" * 40)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test,y_pred,target_names=["Fail", "Pass"]))

results = X_test.copy()
results["Actual_Result"] = y_test.values
results["Predicted_Result"] = y_pred
results["Pass_Probability"] = y_probability

results["Actual_Result"] = results["Actual_Result"].map({0: "Fail",1: "Pass"})
results["Predicted_Result"] = results["Predicted_Result"].map({0: "Fail",1: "Pass"})
print("\nActual and Predicted Results:")
print(results.round(4))
study_hours = float(input("\nEnter study hours per day: "))
attendance = float(input("Enter attendance percentage: "))

new_student = pd.DataFrame({"Study_Hours": [study_hours],"Attendance": [attendance]})

new_student_scaled = scaler.transform(new_student)

prediction = model.predict(new_student_scaled)[0]
pass_probability = model.predict_proba(new_student_scaled)[0][1]

if prediction == 1:
    print("\nPredicted Result: PASS")
else:
    print("\nPredicted Result: FAIL")
print(f"Probability of Passing: "
    f"{pass_probability * 100:.2f}%")
