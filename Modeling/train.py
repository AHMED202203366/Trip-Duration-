import joblib 
import pandas as pd
from preprocessing import *
from model import make_pipeline, eval_model

if __name__ == "__main__":
    # 1. Data Loading & Path Configuration
    data_paths = {
        'train': r'D:\Course Mostafa Saad\HomeWork & Projects\Projects\Taxi_Trip\Data\split\split\train.csv',
        'val': r'D:\Course Mostafa Saad\HomeWork & Projects\Projects\Taxi_Trip\Data\split\split\val.csv'
    }

    # Load datasets using dictionary comprehension (more concise)
    datasets = {
        name: pd.read_csv(path) 
        for name, path in data_paths.items()
    }

    # 2. Data Preprocessing
    for name in datasets:
        datasets[name] = preprocess_data(datasets[name])

    # 3. Feature Configuration
    numeric_columns = [
        'pickup_latitude', 'pickup_longitude', 
        'dropoff_latitude', 'dropoff_longitude',
        'hour', 'weekday', 'is_weekend', 
        'month', 'distance', 'is_peak_hour', 'hour_weekday_interaction',
        'distance_short', 'dirction', 'mnhattan_short_path'
    ]
    categorical_columns = ['time_of_day']
    all_features = numeric_columns + categorical_columns
    target_column = 'trip_duration_log'

    # 4. Model Training
    model = make_pipeline(numeric_columns, categorical_columns)
    model.fit(datasets['train'][all_features], datasets['train'][target_column])
    
    
    # 5. Evaluation & Reporting
    for dataset_name in ['train', 'val']:
        print(f"\n{dataset_name.capitalize()} Data Evaluation:")
        eval_model(
            model,
            datasets[dataset_name][all_features],
            datasets[dataset_name][target_column]
        )
    print("\nModel evaluation complete.")
    print('__________________________________')

    # Save model to a pickle file  
    # MODEL_PATH = r"D:\Course Mostafa Saad\HomeWork & Projects\Taxi_Trip\Modeling/model.pkl"   
    MODEL_PATH = r'D:\Course Mostafa Saad\HomeWork & Projects\Projects\Taxi_Trip\Modeling\model.pkl'
    joblib.dump(model, MODEL_PATH)

    print("\nModel Saving complete.")

    
