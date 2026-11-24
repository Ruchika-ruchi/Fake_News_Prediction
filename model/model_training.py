import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load training data
with open("../data/training_data.pkl", "rb") as file:
    data = pickle.load(file)

X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("Logistic Regression Results")
print("Accuracy:", accuracy * 100, "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save trained model
with open("logistic_model.pkl", "wb") as file:
    pickle.dump(model, file)
print("\nModel saved successfully!")