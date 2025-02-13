import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from preprocessing import *

# Function to evaluate the model performance
def eval_model(model, x, target):
    """
    Computes RMSE and R2 score for the given model.
    
    Parameters:
        model: Trained machine learning model.
        x: Feature set.
        target: True target values.
    
    Returns:
        rmse: Root Mean Squared Error.
        r2: R2 score.
    """
    y_predict = model.predict(x)
    rmse = np.sqrt(mean_squared_error(target, y_predict))
    r2 = r2_score(target, y_predict)
    print(f'RMSE = {rmse:.4f} and R2 score = {r2:.4f}')
    return rmse, r2

# Function to apply log transformation to numeric features
def log_transform(x):
    """
    Applies log1p transformation to numeric features to handle skewness.
    
    Parameters:
        x: Numeric feature array.
    
    Returns:
        Transformed feature array.
    """
    return np.log1p(np.maximum(x, 0))

# Function to create the model pipeline
def make_pipeline(numeric_columns, categorical_columns):
    """
    Builds a machine learning pipeline including preprocessing and Ridge regression.
    
    Parameters:
        numeric_columns: List of numeric feature column names.
        categorical_columns: List of categorical feature column names.
    
    Returns:
        model: Sklearn pipeline object.
    """
    LogFeatures = FunctionTransformer(log_transform)
    
    # Pipeline for numeric feature transformation
    numeric_transform = Pipeline(steps=[ 
        ('scaler', StandardScaler()),  # Standardizing numerical features
        ('poly', PolynomialFeatures(degree=3, include_bias=False)),  # Adding polynomial features
        ('log', LogFeatures)  # Applying log transformation
    ])
    
    # Pipeline for categorical feature transformation
    categorical_transform = Pipeline(steps=[ 
        ('ohe', OneHotEncoder(handle_unknown='ignore'))  # One-hot encoding categorical variables
    ])
    
    # Combining transformations
    transformer = ColumnTransformer(
        transformers=[
            ('num', numeric_transform, numeric_columns),
            ('cat', categorical_transform, categorical_columns)
        ]
    )
    
    # Full pipeline with preprocessing and regression model
    model = Pipeline(steps=[
        ('preprocessor', transformer),
        ('regressor', Ridge(alpha=1.0))  # Using Ridge Regression
    ])
    
    return model
