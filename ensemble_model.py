import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
train_df = pd.read_csv('filtered_train.csv')
test_df = pd.read_csv('filtered_test.csv')

# Split features and target variable
X_train = train_df.drop(columns=['Depression'])
y_train = train_df['Depression']
X_test = test_df.drop(columns=['Depression'], errors='ignore')

# Use a subset of data for faster training
X_train_sample = X_train.sample(n=5000, random_state=42)
y_train_sample = y_train.loc[X_train_sample.index]

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sample)
X_test_scaled = scaler.transform(X_test)

# Define base classifiers with optimized hyperparameters
rf_clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1,
                                warm_start=True)  # Reduce n_estimators for speed
lr_clf = LogisticRegression(random_state=42, max_iter=1000, solver='saga',
                            n_jobs=-1)  # saga solver is efficient for large datasets
svc_clf = LinearSVC(random_state=42)  # Faster than standard SVC

# Create a voting classifier
ensemble_model = VotingClassifier(estimators=[
    ('random_forest', rf_clf),
    ('logistic_regression', lr_clf),
    ('svc', svc_clf)
], voting='hard', n_jobs=-1)  # Use 'hard' voting to avoid probability computation overhead

# Train the model
ensemble_model.fit(X_train_scaled, y_train_sample)

# Evaluate on training set
y_train_pred = ensemble_model.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train_sample, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_train_sample, y_train_pred))

# Predict on test set
y_pred = ensemble_model.predict(X_test_scaled)

test_df['Depression'] = y_pred
test_df.to_csv('submission_2.csv', index=False)
print("Prediction completed and saved to 'submission_2.csv'")

# Confusion matrix
conf_matrix = confusion_matrix(y_train_sample, y_train_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Depression', 'Depression'],
            yticklabels=['No Depression', 'Depression'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Feature importance visualization (using Random Forest feature importance)
# Extract the trained RandomForestClassifier from VotingClassifier
rf_fitted = ensemble_model.named_estimators_['random_forest']

if hasattr(rf_fitted, 'feature_importances_'):
    feature_importances = rf_fitted.feature_importances_
    feature_names = X_train_sample.columns  # Ensure correct feature names

    # Create DataFrame for visualization
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)  # Top 10 features

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, hue='Feature', palette='viridis', legend=False)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature Name")
    plt.title("Top 10 Important Features")
    plt.show()

else:
    print("Feature importance attribute not found in the model.")





