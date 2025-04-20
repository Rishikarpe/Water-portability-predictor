# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import joblib  # For saving models

# Load the dataset
df = pd.read_csv('dataset.csv')

# Check for missing values and impute with the mean
df.fillna(df.mean(), inplace=True)

# Split the data into features and target variable
X = df.drop('Potability', axis=1)  # Features
y = df['Potability']  # Target variable

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling for models sensitive to feature range (Logistic Regression, KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the models
lr_model = LogisticRegression()
knn_model = KNeighborsClassifier()
rf_model = RandomForestClassifier()

# Train the models
lr_model.fit(X_train_scaled, y_train)
knn_model.fit(X_train_scaled, y_train)
rf_model.fit(X_train_scaled, y_train)

# Evaluate the models
lr_accuracy = lr_model.score(X_test_scaled, y_test)
knn_accuracy = knn_model.score(X_test_scaled, y_test)
rf_accuracy = rf_model.score(X_test_scaled, y_test)

# Print accuracy of the models
print("Logistic Regression Accuracy:", lr_accuracy)
print("K-Nearest Neighbors Accuracy:", knn_accuracy)
print("Random Forest Accuracy:", rf_accuracy)

# Confusion Matrix for Random Forest
y_pred_rf = rf_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix:\n", cm)

# Classification Report for Random Forest
report = classification_report(y_test, y_pred_rf)
print("Classification Report:\n", report)

# Plot ROC Curve for Random Forest
fpr, tpr, thresholds = roc_curve(y_test, rf_model.predict_proba(X_test_scaled)[:, 1])
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Random Forest (AUC = %0.2f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Feature Importance from Random Forest
feature_importance = rf_model.feature_importances_
features = X.columns
plt.barh(features, feature_importance)
plt.xlabel("Feature Importance")
plt.title("Feature Importance for Water Potability Prediction")
plt.show()

# Save the models and scaler using joblib
joblib.dump(rf_model, 'random_forest_model.pkl')  # Save Random Forest model
joblib.dump(lr_model, 'logistic_regression_model.pkl')  # Save Logistic Regression model
joblib.dump(knn_model, 'knn_model.pkl')  # Save KNN model
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler
