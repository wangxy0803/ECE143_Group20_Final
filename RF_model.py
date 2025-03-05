import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load the data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
submission_df = pd.read_csv('sample_submission.csv')

#1.Data Cleaning and Processing
numeric_cols = [col for col in train_df.select_dtypes(include=['float64', 'int64']).columns if col in test_df.columns]
# Fill missing values for numeric columns
train_df[numeric_cols] = train_df[numeric_cols].fillna(train_df[numeric_cols].mean())
test_df[numeric_cols] = test_df[numeric_cols].fillna(test_df[numeric_cols].mean())

# Fill missing values for categorical columns
categorical_cols = train_df.select_dtypes(include=['object']).columns.intersection(test_df.columns)
for col in categorical_cols:
    train_df[col].fillna(train_df[col].mode()[0], inplace=True)
    test_df[col].fillna(test_df[col].mode()[0], inplace=True)

# Encode categorical features
train_df = pd.get_dummies(train_df, drop_first=True)
test_df = pd.get_dummies(test_df, drop_first=True)
# Ensure both datasets have the same columns
test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

# Separate features and target
X = train_df.drop(['id', 'Depression'], axis=1)
y = train_df['Depression']
# Split the data into training and validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#2.Train the RF model
# Initialize the model
rf = RandomForestClassifier(random_state=42)
# Train the model
rf.fit(X_train, y_train)

#3.Evaluate the model
# Predictions
y_pred = rf.predict(X_val)
# Evaluation metrics
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))

#4.Hyperparameter Tuning with Grid Search
# Define parameter grid with 'sqrt' for max_features instead of 'auto'
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt']
}
# Initialize GridSearchCV with more detailed verbosity
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=3, scoring='accuracy')
# Fit the grid search
grid_search.fit(X_train, y_train)
# Print best parameters and score after the search completes
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)
# Convert cv_results_ to a DataFrame for better readability
cv_results = pd.DataFrame(grid_search.cv_results_)
# Display results of each parameter combination tried
for i, row in cv_results.iterrows():
    print(f"Iteration {i+1}")
    print(f"Mean Accuracy: {row['mean_test_score']:.4f}")
    print(f"Std Dev Accuracy: {row['std_test_score']:.4f}")
    print(f"Parameters: {row['params']}")
    print("-" * 40)

#5. Retrain the Model with the Best Parameters
# Initialize the model with the best parameters
best_rf = RandomForestClassifier(**grid_search.best_params_, random_state=42)
# Retrain the model
best_rf.fit(X_train, y_train)
# Evaluate on the validation set
y_val_pred = best_rf.predict(X_val)
print("Tuned Model Accuracy:", accuracy_score(y_val, y_val_pred))
print("Classification Report:\n", classification_report(y_val, y_val_pred))

#6. Make Predictions on the Test Set
# Prepare test features by removing 'id' and 'Depression' columns
X_test = test_df.drop(['id', 'Depression'], axis=1, errors='ignore')
# Predict using the tuned model
test_predictions = best_rf.predict(X_test)

#7. Submission file
# Prepare submission DataFrame
submission_df['Depression'] = test_predictions
submission_df.to_csv('final_submission.csv', index=False)

#8.Visualization on prediction
# Feature Importance Plot
# Extract Feature Importance
feature_importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': best_rf.feature_importances_})
feature_importances = feature_importances.sort_values(by="Importance", ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importances[:10],hue="Feature", palette="viridis",dodge=False, legend=False)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Top 10 Key Drivers of Mental Health Issues")
plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Depression", "Depression"], yticklabels=["No Depression", "Depression"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()