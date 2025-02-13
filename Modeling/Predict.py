import joblib
import pandas as pd
from preprocessing import preprocess_data
from model import eval_model

test_path = r'D:\Course Mostafa Saad\HomeWork & Projects\Projects\Taxi_Trip\Data\split\split\test.csv'
MODEL_PATH = r'D:\Course Mostafa Saad\HomeWork & Projects\Projects\Taxi_Trip\Modeling\model.pkl'

def predict():
    """Load model, preprocess test data, and make predictions."""
    
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
    
    df_test = pd.read_csv(test_path)
    test_data = preprocess_data(df_test)

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

    rmse, r2 = eval_model(model, test_data[all_features], test_data[target_column])
    
if __name__ == "__main__":
    predict()
