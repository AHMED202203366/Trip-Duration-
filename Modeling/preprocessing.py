import numpy as np
import pandas as pd
import math
from geopy import distance
from geopy.point import Point

# Function to categorize time of day
def categorize_time_of_day(hour):
    """
    Categorizes the hour of the day into morning, afternoon, evening, or night.

    Parameters:
        hour (int): The hour of the day (0-23).

    Returns:
        str: The corresponding time of day category.
    """
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 22:
        return 'evening'
    return 'night'  # Default case for hours between 22:00 - 04:59

# Function to calculate the Haversine distance between two geographic points
def calculate_haversine(lat1, lon1, lat2, lon2):
    """
    Computes the Haversine distance (great-circle distance) between two geographic points.

    Parameters:
        lat1, lon1 (float or array-like): Latitude and longitude of the first point.
        lat2, lon2 (float or array-like): Latitude and longitude of the second point.

    Returns:
        float or np.array: Haversine distance in kilometers.
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Compute differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    # Radius of the Earth in kilometers (6371 km)
    return 6371 * c

def get_distance_km(df):
    """
    Computes the geodesic (great-circle) distance between pickup and dropoff coordinates.

    Parameters:
        df (pd.DataFrame): The DataFrame containing latitude and longitude of pickup and dropoff locations.

    Returns:
        float: Haversine distance in kilometers.
    """
    pickup_coords = (df['pickup_latitude'], df['pickup_longitude'])
    dropoff_coords = (df['dropoff_latitude'], df['dropoff_longitude'])
    distance_km = distance.geodesic(pickup_coords, dropoff_coords).km
    return distance_km

def calculate_direction(row):
    """
    Calculates the bearing direction (angle) between pickup and dropoff locations.

    Parameters:
        row (pd.Series): A row of the DataFrame containing pickup and dropoff coordinates.

    Returns:
        float: Bearing direction in degrees (0-360).
    """
    pickup_coordinates = Point(row['pickup_latitude'], row['pickup_longitude'])
    dropoff_coordinates = Point(row['dropoff_latitude'], row['dropoff_longitude'])
    delta_longitude = dropoff_coordinates[1] - pickup_coordinates[1]
    y = math.sin(math.radians(delta_longitude)) * math.cos(math.radians(dropoff_coordinates[0]))
    x = math.cos(math.radians(pickup_coordinates[0])) * math.sin(math.radians(dropoff_coordinates[0])) - \
        math.sin(math.radians(pickup_coordinates[0])) * math.cos(math.radians(dropoff_coordinates[0])) * \
        math.cos(math.radians(delta_longitude))
    bearing = math.atan2(y, x)
    bearing = math.degrees(bearing)
    bearing = (bearing + 360) % 360
    return bearing

def manhattan_distance(df):
    """
    Calculates the Manhattan distance between pickup and dropoff locations.

    Parameters:
        df (pd.Series): A row of the DataFrame containing pickup and dropoff latitude/longitude.

    Returns:
        float: Manhattan distance in kilometers.
    """
    lat_distance = abs(df['pickup_latitude'] - df['dropoff_latitude']) * 111  
    lon_distance = abs(df['pickup_longitude'] - df['dropoff_longitude']) * 111 * math.cos(math.radians(df['pickup_latitude']))
    return lat_distance + lon_distance

def preprocess_data(df):
    """
    Preprocesses the taxi trip dataset by extracting relevant features.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing trip data.

    Returns:
        pd.DataFrame: Processed DataFrame with new features.
    """
    # Convert pickup_datetime to datetime format (vectorized for efficiency)
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')

    # Log-transform trip_duration (to reduce skewness)
    df['trip_duration_log'] = np.log1p(df['trip_duration'])

    # Compute Haversine distance for all rows using vectorized operations
    df['distance'] = calculate_haversine(
        df['pickup_latitude'].values, df['pickup_longitude'].values,
        df['dropoff_latitude'].values, df['dropoff_longitude'].values
    )

    # Extract date-related features
    df['weekday'] = df['pickup_datetime'].dt.weekday         # Day of the week (0=Monday, 6=Sunday)
    df['hour'] = df['pickup_datetime'].dt.hour               # Hour of the day (0-23)
    df['month'] = df['pickup_datetime'].dt.month             # Month of the year (1-12)
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)      # Binary flag for weekends

    # Categorize time of day efficiently using vectorized apply
    df['time_of_day'] = np.select(
        [df['hour'].between(5, 11), df['hour'].between(12, 16), df['hour'].between(17, 21)],
        ['morning', 'afternoon', 'evening'],
        default='night'
    )

    # Adding peak hour feature
    df['is_peak_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)

    # Interaction terms (hour and weekday interaction)
    df['hour_weekday_interaction'] = df['hour'] * df['weekday']

    # Drop unnecessary columns to reduce memory usage
    df.drop(columns=['pickup_datetime', "id"], inplace=True, errors='ignore')

    df['distance_short'] = df.apply(get_distance_km, axis=1)
    df['dirction'] = df.apply(calculate_direction, axis=1)
    df['mnhattan_short_path'] = df.apply(manhattan_distance, axis=1)

    return df