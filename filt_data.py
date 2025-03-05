import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.ensemble import IsolationForest

#load data
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")


#(1)Remove unrelated columns from the dataset
columns_to_drop_ID = ['id', 'Name','Working Professional or Student', 'CGPA']
df_train = df_train.drop(columns=columns_to_drop_ID)
df_test = df_test.drop(columns=columns_to_drop_ID)
df_train['Profession'] = df_train['Profession'].fillna('Student')
df_test['Profession'] = df_test['Profession'].fillna('Student')

#(2)Merge similar columns
df_train['Pressure'] = df_train['Work Pressure'].combine_first(df_train['Academic Pressure'])
df_test['Pressure'] = df_test['Work Pressure'].combine_first(df_test['Academic Pressure'])
df_train = df_train.drop(columns=['Work Pressure', 'Academic Pressure'])
df_test = df_test.drop(columns=['Work Pressure', 'Academic Pressure'])

df_train['Satisfaction'] = df_train['Job Satisfaction'].combine_first(df_train['Study Satisfaction'])
df_test['Satisfaction'] = df_test['Job Satisfaction'].combine_first(df_test['Study Satisfaction'])
df_train = df_train.drop(columns=['Job Satisfaction', 'Study Satisfaction'])
df_test = df_test.drop(columns=['Job Satisfaction', 'Study Satisfaction'])

#(3)Map the contents of non-numeric columns to numeric values.

# Target encoding for categorical features
encoder = TargetEncoder(cols=['City', 'Profession'])
df_train[['City_encoded', 'Profession_encoded']] = encoder.fit_transform(df_train[['City', 'Profession']], df_train["Depression"])
df_test[['City_encoded', 'Profession_encoded']] = encoder.transform(df_test[['City', 'Profession']])

# Define features and target
X_train = df_train.drop('Depression', axis=1)
y_train = df_train['Depression']

# Redefine columns for preprocessing after feature engineering
numerical_columns = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()
transformed_columns = numerical_columns + categorical_columns

# Define preprocessing pipelines
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('convert_to_float32', FunctionTransformer(lambda x: x.astype(np.float32)))
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('ordinal', OrdinalEncoder(dtype=np.int32, handle_unknown='use_encoded_value', unknown_value=-1))
])

# Combine the numerical and categorical pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_columns),
        ('cat', categorical_pipeline, categorical_columns)
    ]
)

# Apply the transformations to the training and test sets
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(df_test)

# Apply Isolation Forest for outlier detection on the training data
isolation_forest = IsolationForest(contamination=0.04, random_state=10)
outlier_labels = isolation_forest.fit_predict(X_train_preprocessed)

# Filter out outliers from both X_train_preprocessed and y_train
non_outliers_mask = outlier_labels != -1
X_train_preprocessed = pd.DataFrame(X_train_preprocessed, columns=transformed_columns)
X_train_preprocessed = X_train_preprocessed[non_outliers_mask]
y_train = y_train[non_outliers_mask]

X_test_preprocessed = pd.DataFrame(X_test_preprocessed, columns=transformed_columns)
X_train_preprocessed['Depression'] = y_train

filtered_file_path1 = 'filtered_train.csv'
filtered_file_path2 = 'filtered_test.csv'
X_train_preprocessed.to_csv(filtered_file_path1, index=False)
X_test_preprocessed.to_csv(filtered_file_path2, index=False)
