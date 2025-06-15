from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data: marks of students
# [math_marks, science_marks]
X = [
    [85, 90],
    [45, 50],
    [60, 65],
    [30, 40],
    [75, 80],
    [35, 25],
    [90, 95],
    [50, 55],
    [20, 30],
    [70, 60]
]

# 1 = Pass, 0 = Fail
y = [1, 0, 1, 0, 1, 0, 1, 1, 0, 1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc*100:.2f}%")

# Try a custom prediction
marks = [[55, 60]]  # try changing this
prediction = model.predict(marks)
print("Prediction for marks [55, 60]:", "Pass" if prediction[0]==1 else "Fail")
