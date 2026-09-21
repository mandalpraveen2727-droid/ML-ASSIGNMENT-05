# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,confusion_matrix,roc_curve,roc_auc_score)

# 1. Create the custom student dataset
data = {
    "Study_Hours": [
        1.0, 1.5, 2.0, 2.5, 3.0,3.5, 4.0, 4.5, 5.0, 5.5,
        6.0, 6.5, 7.0, 7.5, 8.0,8.5, 9.0, 2.0, 4.0, 6.0,
        3.0, 5.0, 7.0, 8.0, 1.5,4.5, 5.5, 6.5, 7.5, 9.5],

    "Attendance": [
        45, 50, 55, 58, 60,62, 65, 68, 72, 75,
        78, 80, 82, 85, 88,90, 92, 48, 55, 65,
        52, 68, 75, 80, 40,60, 70, 73, 83, 95],

    # 0 represents Fail and 1 represents Pass
    "Result": [
        0, 0, 0, 0, 0,0, 0, 1, 1, 1,
        1, 1, 1, 1, 1,1, 1, 0, 0, 1,
        0, 1, 1, 1, 0,0, 1, 1, 1, 1]
}

# Convert the dictionary into a DataFrame
df = pd.DataFrame(data)
# 2. Display dataset information
print("Student Dataset:")
print(df)

print("\nDataset Shape:")
print(df.shape)

print("\nClass Distribution:")
print(df["Result"].value_counts())
# 3. Separate features and target

X = df[["Study_Hours","Attendance"]]
y = df["Result"]

# 4. Divide data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split( X, y,test_size=0.30, random_state=42,stratify=y)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

# 5. Scale the input features
scaler = StandardScaler()

# Learn scaling values from training data
X_train_scaled = scaler.fit_transform(X_train)

# Apply the same transformation to testing data
X_test_scaled = scaler.transform(X_test)
# 6. Create and train Logistic Regression model

model = LogisticRegression(random_state=42)
model.fit(X_train_scaled,y_train)

# 7. Predict classes
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test,y_pred)
print("\nModel Performance:")
print(f"Accuracy: {accuracy:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 8. Calculate probabilitie
# Get probability of class 1, which represents Pass
pass_probabilities = model.predict_proba(X_test_scaled)[:, 1]
print("\nPass Probabilities:")
print(pass_probabilities.round(4))

# 9. Calculate ROC curve
false_positive_rate, true_positive_rate, thresholds = (
    roc_curve(y_test,pass_probabilities))
# 10. Calculate AUC score

auc_score = roc_auc_score(y_test,pass_probabilities)
print(f"\nAUC Score: {auc_score:.4f}")

# 11. Display ROC values
roc_values = pd.DataFrame({"Threshold": thresholds,"False Positive Rate": false_positive_rate,"True Positive Rate": true_positive_rate})

print("\nROC Values:")
print(roc_values.round(4))

# 12. Display actual and predicted results
results = X_test.copy()
results["Actual_Result"] = y_test.values
results["Predicted_Result"] = y_pred
results["Pass_Probability"] = pass_probabilities

# Convert numerical result into meaningful labels
results["Actual_Result"] = results["Actual_Result"
].map({0: "Fail",1: "Pass"})
results["Predicted_Result"] = results["Predicted_Result"].map({0: "Fail",1: "Pass"})
print("\nActual and Predicted Results:")
print(results.round(4))

# 13. Plot the ROC curve
plt.figure(figsize=(8, 6))

# Plot the model's ROC curve
plt.plot(false_positive_rate,true_positive_rate,color="blue",linewidth=2,marker="o",label=f"Logistic Regression (AUC = {auc_score:.4f})")

# Plot the random-classifier reference line
plt.plot([0, 1],[0, 1], color="red",linewidth=2,linestyle="--",label="Random Classifier (AUC = 0.5)")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Student Pass/Fail Prediction")
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()