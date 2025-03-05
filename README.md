# README

## Project Overview
This project focuses on analyzing a mental health dataset and training machine learning models to predict depression. The workflow consists of two main steps:
1. **Data Analysis:** Finding correlations, trends, and interesting groupings. Visualizing the data to see patterns and outliers.
2. **Model Training and Prediction:** Using 3 different models make predictions: Random Forest, a Voting Classifier composed of Random Forest, Logistic Regression, and LinearSVC after data filtring, and a Stacking Ensemble Model combined with XGBoost, CatBoost and HistGradientBoosting.

## File Structure
```
.
├── train.csv                         # Raw training dataset
├── test.csv                          # Raw test dataset
├── data_analysis.ipynb                          # Data analysis and visualization
├── RF_model.py                     # Script for training the first model and making predictions
├── filtered_train.csv                 # Preprocessed training dataset before second model
├── filtered_test.csv                  # Preprocessed test dataset before second model
├── filt_data.py                      # Script for data preprocessing
├── ensemble_model.py                     # Script for training the second model and making predictions
├── stacking_ensemble_model.py                     # Script for training the third model and making predictions
├── submission_1.csv          # Final output file with predictions using the first model
├── submission_2.csv          # Final output file with predictions using the second model
├── submission_3.csv          # Final output file with predictions using the third model
├── README.md                          # Documentation
```
## Requirements

- Python 3.7+ (or higher)
- [NumPy](https://pypi.org/project/numpy/)
- [pandas](https://pypi.org/project/pandas/)
- [scikit-learn](https://pypi.org/project/scikit-learn/) (>= 1.2 recommended)
- [CatBoost](https://pypi.org/project/catboost/)
- [XGBoost](https://pypi.org/project/xgboost/)
- [matplotlib](https://pypi.org/project/matplotlib/) or [seaborn](https://pypi.org/project/seaborn/) (if visualization is included)

You can install them via `pip`:

```bash
pip install numpy pandas scikit-learn catboost xgboost matplotlib seaborn
```
Or via `conda`:

```bash
conda install -c conda-forge numpy pandas scikit-learn catboost xgboost matplotlib seaborn
```

## Installation and Model Details
### Model 1
Run the model training script:
```bash
python RF_model.py
```
The model is a Random Forest Classifier with Grid Search to tune the hyperparameters.
### Model 2
#### Step 1: Preprocess the Data
Run the following command to clean and preprocess the dataset:
```sh
python filt_data.py
```
This will generate `filtered_train.csv` and `filtered_test.csv`, which are the cleaned datasets.

#### Step 2: Train the Model and Make Predictions
Run the model training script:
```sh
python ensemble_model.py
```
This will train the model using the processed dataset and save predictions in `submission_2.csv`.

The model is a Voting Classifier that combines:
- **RandomForestClassifier**: n_estimators=50 (reduced for speed optimization)
- **LogisticRegression**: Solver='saga' with max_iter=1000
- **LinearSVC**: A fast linear SVM implementation

Data is standardized using `StandardScaler`, and an `IsolationForest` is applied to remove outliers before training the model.

### Model 3
Run the model training script:
```sh
python stacking_ensemble_model.py
```
This will train the model using the processed dataset and save predictions in `submission_3.csv`.

The model is a Ensemble Classifier that combines:
- **XGBoost** and **CatBoost** as individual classifiers
- **Stacking Ensemble** to combine multiple models
- Use `cross_val_score` with `scoring=accuracy_score` to show cross-validation results

## Notes
- The scripts assume `train.csv` and `test.csv` are available in the working directory.
- The dataset includes categorical and numerical features, which are handled using different preprocessing techniques.

If you encounter any issues, ensure all dependencies are installed and that the input files are correctly formatted.
