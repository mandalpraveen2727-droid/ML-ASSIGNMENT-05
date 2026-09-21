import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# --------------------------------------------------
# 1. Create the custom student dataset
# --------------------------------------------------

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

    # 0 = Fail and 1 = Pass
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

# --------------------------------------------------
# 2. Separate features and target
# --------------------------------------------------

X = df[["Study_Hours", "Attendance"]]
y = df["Result"]

# --------------------------------------------------
# 3. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 4. Feature scaling
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# 5. Train Logistic Regression model
# --------------------------------------------------

model = LogisticRegression(
    random_state=42
)

model.fit(
    X_train_scaled,
    y_train
)

# --------------------------------------------------
# 6. Calculate probabilities
# --------------------------------------------------

# Probability that a student will pass
pass_probabilities = model.predict_proba(
    X_test_scaled
)[:, 1]

print("\nPass Probabilities:")
print(pass_probabilities.round(4))

# --------------------------------------------------
# 7. Apply different decision thresholds
# --------------------------------------------------

thresholds = [0.3, 0.5, 0.7]

comparison_results = []

for threshold in thresholds:

    # Convert probability into class using the threshold
    y_pred = (
        pass_probabilities >= threshold
    ).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # Store results
    comparison_results.append({
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    # Display individual threshold result
    print(f"\nResults for Threshold = {threshold}")
    print("-" * 40)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# --------------------------------------------------
# 8. Display comparison table
# --------------------------------------------------

comparison_df = pd.DataFrame(
    comparison_results
)

print("\nThreshold Comparison:")
print(comparison_df.round(4))

# --------------------------------------------------
# 9. Display predictions for every threshold
# --------------------------------------------------

prediction_table = X_test.copy()

prediction_table["Actual_Result"] = y_test.values
prediction_table["Pass_Probability"] = pass_probabilities

for threshold in thresholds:
    prediction_table[
        f"Prediction_{threshold}"
    ] = (
        pass_probabilities >= threshold
    ).astype(int)

print("\nPredictions at Different Thresholds:")
print(prediction_table.round(4))