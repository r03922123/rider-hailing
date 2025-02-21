#!/usr/bin/env python
# coding: utf-8

# In[304]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[305]:


df = pd.read_csv("robotex5.csv")
df["start_time"] = pd.to_datetime(df["start_time"])

df.head()


# In[306]:


# 1. Basic data info
print("Basic Data Info:")
df.info()


# In[307]:


# 2. Basic statistics
print("\nNumerical Statistics:")
print(df.describe())


# ## Data Cleasing

# In[308]:


# Remove rows with missing values
print("\nNumber of rows before removing missing values:", len(df))
df = df.dropna()
print("Number of rows after removing missing values:", len(df))

# Remove duplicates
print("\nNumber of rows before removing duplicates:", len(df))
df = df.drop_duplicates()
print("Number of rows after removing duplicates:", len(df))


# Check types of columns
# all value in start_time column should be datetime
for col in df.columns:
    if col == "start_time":
        assert all(isinstance(x, pd.Timestamp) for x in df[col])
    else:
        assert all(isinstance(x, float) for x in df[col])


# ## EDA

# In[309]:


EUROPE_BOUNDS = {
    "lat_min": 35.0,  # Southernmost point
    "lat_max": 71.0,  # Northernmost point
    "lng_min": -25.0,  # Westernmost point
    "lng_max": 45.0,  # Easternmost point
}
# Check data with any of start_lat, start_lng, end_lat, end_lng outside of Europe
print("\nData with any of start_lat, start_lng, end_lat, end_lng outside of Europe:")
df_ooeu = df[
    (df["start_lat"] < EUROPE_BOUNDS["lat_min"])
    | (df["start_lat"] > EUROPE_BOUNDS["lat_max"])
    | (df["start_lng"] < EUROPE_BOUNDS["lng_min"])
    | (df["start_lng"] > EUROPE_BOUNDS["lng_max"])
    | (df["end_lat"] < EUROPE_BOUNDS["lat_min"])
    | (df["end_lat"] > EUROPE_BOUNDS["lat_max"])
    | (df["end_lng"] < EUROPE_BOUNDS["lng_min"])
    | (df["end_lng"] > EUROPE_BOUNDS["lng_max"])
]
print(df_ooeu)


# In[310]:


from geopy.geocoders import Nominatim


def get_country(lat, lng):
    try:
        location = geolocator.reverse(f"{lat}, {lng}")
        if not location:
            return None
        address = location.raw["address"]
        return address.get("country", None)
    except Exception as e:
        print(f"Error: {e}")
        return None


def check_order(df):
    for i, row in df.iterrows():
        print(f"ride_value: {row['ride_value']}")
        start_country_code = get_country(row["start_lat"], row["start_lng"])
        end_country_code = get_country(row["end_lat"], row["end_lng"])
        print(f"Row {i}: {start_country_code} -> {end_country_code}")
        print("-" * 50)


# Sample df_ooeu and check it's start and end country
geolocator = Nominatim(user_agent="robotex5")
sample_ooeu = df_ooeu.sample(10)
# check_order(sample_ooeu)


# In[311]:


# remove df_ooeu from df
print("\nNumber of rows before removing data outside of Europe:", len(df))
df = df.drop(df_ooeu.index)
print("Number of rows after removing data outside of Europe:", len(df))


# In[312]:


# Visualizations
plt.figure(figsize=(15, 10))

# Pickup locations
plt.subplot(221)
plt.scatter(df["start_lng"], df["start_lat"], alpha=0.5)
plt.title("Pickup Locations")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# Dropoff locations
plt.subplot(222)
plt.scatter(df["end_lng"], df["end_lat"], alpha=0.5, color="red")
plt.title("Dropoff Locations")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# Ride value distribution
plt.subplot(223)
sns.histplot(df["ride_value"], bins=50)
plt.title("Distribution of Ride Values")
plt.xlabel("Ride Value")
plt.ylabel("Count")

# Time analysis
df["hour"] = df["start_time"].dt.hour

plt.subplot(224)
sns.boxplot(x="hour", y="ride_value", data=df)
plt.title("Ride Values by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Ride Value")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# In[313]:


plt.figure(figsize=(15, 10))

# Original distribution
plt.subplot(121)
sns.boxplot(y=df["ride_value"])
plt.title("Boxplot of Ride Values\n(with outliers)")
plt.ylabel("Ride Value")

plt.subplot(122)
# Make count as log scale in histogram
g = sns.histplot(df["ride_value"], bins=50)
g.set_yscale("log")

plt.title("Distribution of Ride Values\n(Log Scale)")
plt.xlabel("Ride Value")
plt.ylabel("Count")

# # Distribution without outliers
# plt.subplot(133)
# sns.histplot(data=df[df["ride_value"] <= upper_bound], x="ride_value", bins=50)
# plt.title(f"Distribution of Ride Values\n(≤ {upper_bound:.2f})")
# plt.xlabel("Ride Value")
# plt.ylabel("Count")

plt.tight_layout()
plt.show()


# In[314]:


# Basic statistics of ride_value
print("Ride Value Statistics:")
print(df["ride_value"].describe([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))


# In[315]:


# Calculate potential outliers
# naive way to set upper bound
upper_bound = 100

outliers = df[df["ride_value"] > upper_bound]
print(f"\nNumber of outliers: {len(outliers)}")
print(f"Percentage of outliers: {(len(outliers) / len(df)) * 100:.2f}%")
print("\nTop 10 highest ride values:")
print(df["ride_value"].nlargest(10))

# print("\nSample 10 data from outliers:")
# sample_outlier = outliers.sample(10)
# check_order(sample_outlier)


# In[316]:


# Add time components
df["hour"] = df["start_time"].dt.hour
df["day_of_week"] = df["start_time"].dt.day_name()
# df["date"] = df["start_time"].dt.date

# Create outlier flag
df["is_outlier"] = df["ride_value"] > upper_bound

# 1. Spatial Distribution
plt.figure(figsize=(15, 10))

# Pickup locations
plt.subplot(221)
plt.scatter(
    df[~df["is_outlier"]]["start_lng"],
    df[~df["is_outlier"]]["start_lat"],
    alpha=0.5,
    label="Normal",
    color="blue",
)
plt.scatter(
    df[df["is_outlier"]]["start_lng"],
    df[df["is_outlier"]]["start_lat"],
    alpha=0.5,
    label="Outlier",
    color="red",
)
plt.title("Pickup Locations (Normal vs Outliers)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()

# Dropoff locations
plt.subplot(222)
plt.scatter(
    df[~df["is_outlier"]]["end_lng"],
    df[~df["is_outlier"]]["end_lat"],
    alpha=0.5,
    label="Normal",
    color="blue",
)
plt.scatter(
    df[df["is_outlier"]]["end_lng"],
    df[df["is_outlier"]]["end_lat"],
    alpha=0.5,
    label="Outlier",
    color="red",
)
plt.title("Dropoff Locations (Normal vs Outliers)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()

# 2. Temporal Distribution
# By hour
plt.subplot(223)
outlier_by_hour = df[df["is_outlier"]]["hour"].value_counts(normalize=True) * 100
normal_by_hour = df[~df["is_outlier"]]["hour"].value_counts(normalize=True) * 100

plt.bar(normal_by_hour.index, normal_by_hour, alpha=0.5, label="Normal", color="blue")
plt.bar(outlier_by_hour.index, outlier_by_hour, alpha=0.5, label="Outlier", color="red")
plt.title("Distribution by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Percentage")
plt.legend()

# By day of week
plt.subplot(224)
outlier_by_day = df[df["is_outlier"]]["day_of_week"].value_counts(normalize=True) * 100
normal_by_day = df[~df["is_outlier"]]["day_of_week"].value_counts(normalize=True) * 100

plt.bar(
    range(len(normal_by_day)), normal_by_day, alpha=0.5, label="Normal", color="blue"
)
plt.bar(
    range(len(outlier_by_day)), outlier_by_day, alpha=0.5, label="Outlier", color="red"
)
plt.title("Distribution by Day of Week")
plt.xticks(range(len(normal_by_day)), normal_by_day.index, rotation=45)
plt.ylabel("Percentage")
plt.legend()

plt.tight_layout()
plt.show()


# In[317]:


df = df[~df["is_outlier"]]


# ## Coordinate Quantization

# In[318]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def prepare_coordinates(df, lat: str, lng: str):
    # Extract start coordinates
    start_coords = df[[lat, lng]].values

    # # Extract end coordinates
    # end_coords = df[["end_lat", "end_lng"]].values

    # Combine unique of start and end coords into a single array
    # all_coords = np.unique(np.vstack((start_coords, end_coords)), axis=0)
    uniq_coords = np.unique(start_coords, axis=0)
    return uniq_coords


# Perform K-means clustering
def perform_kmeans(coordinates, n_clusters=3):
    # Standardize the data
    scaler = StandardScaler()
    scaled_coordinates = scaler.fit_transform(coordinates)

    # Fit K-means
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=3,
        max_iter=300,
        tol=1e-4,
        random_state=42,
    )
    cluster_labels = kmeans.fit_predict(scaled_coordinates)

    # Get cluster centers (transform back to original scale)
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

    return cluster_labels, cluster_centers, scaler, kmeans


# Visualize the results
def visualize_clusters(coordinates, labels, centers):
    plt.figure(figsize=(10, 8))

    # Plot all points
    for cluster_num in range(len(centers)):
        cluster_points = coordinates[labels == cluster_num]
        plt.scatter(
            cluster_points[:, 1],  # longitude
            cluster_points[:, 0],  # latitude
            s=50,
            alpha=0.7,
            label=f"Cluster {cluster_num}",
        )

    # Plot cluster centers
    plt.scatter(
        centers[:, 1],  # longitude
        centers[:, 0],  # latitude
        s=200,
        marker="X",
        c="red",
        label="Centroids",
    )

    plt.title("K-means Clustering of Ride-Hailing Start and End Points")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the figure
    # plt.savefig('tallinn_ride_clusters.png')
    plt.show()


# In[319]:


# Calculate optimal number of clusters using the Elbow method
def find_optimal_clusters(coordinates, min_clusters=8, max_clusters=20, stride=4):
    scaled_coordinates = StandardScaler().fit_transform(coordinates)
    inertias, silhouette_scores = [], []

    for k in range(min_clusters, max_clusters + 1, stride):
        print(f"Calculating for k={k}")
        kmeans = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=3,
            random_state=42,
        )
        cluster_labels = kmeans.fit_predict(scaled_coordinates)
        inertias.append(kmeans.inertia_)
        sample_silhouette_value = silhouette_score(
            scaled_coordinates, cluster_labels, sample_size=int(len(coordinates) * 0.1)
        )
        silhouette_scores.append(sample_silhouette_value)

    return inertias, silhouette_scores


# In[320]:


# Prepare (start) coordinates
all_coords = prepare_coordinates(df, "start_lat", "start_lng")
print(f"Combined coordinates shape: {all_coords.shape}")

min_clusters, max_clusters, stride = 8, 64, 4
inertias, silhouette_scores = find_optimal_clusters(
    all_coords, min_clusters, max_clusters, stride
)
# Plot the elbow curve and silhouette scores
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(range(min_clusters, max_clusters + 1, stride), inertias, "bo-")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.grid(True, alpha=0.3)
# plt.savefig('elbow_method.png')

plt.subplot(1, 2, 2)
plt.plot(range(min_clusters, max_clusters + 1, stride), silhouette_scores, "ro-")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores for k")
plt.grid(True, alpha=0.3)
# plt.savefig('silhouette_scores.png')

plt.show()


# In[321]:


# Perform clustering
n_clusters = 12
labels, centers, scaler, kmeans = perform_kmeans(all_coords, n_clusters)

# Visualize results
visualize_clusters(all_coords, labels, centers)

# Print cluster information
for i, center in enumerate(centers):
    cluster_size = np.sum(labels == i)
    percentage = 100 * cluster_size / len(labels)
    print(f"Cluster {i}:")
    print(f"  Center: Lat {center[0]:.6f}, Lng {center[1]:.6f}")
    print(f"  Size: {cluster_size} points ({percentage:.1f}% of total)")
    print()


# In[322]:


# Create coordinate to cluster hashmap
coord_cluster_map = {
    tuple(coord): cluster for coord, cluster in zip(all_coords, labels)
}


# In[323]:


# Create time series data with start_time, start_alt, start_lng, ride_value
ts_df = df[["start_time", "start_lat", "start_lng", "ride_value"]].copy()

# Create cluster column by coord_cluster_map[(start_lat, start_lng)]
ts_df["cluster"] = ts_df.apply(
    lambda x: coord_cluster_map[(x["start_lat"], x["start_lng"])], axis=1
)


# In[324]:


# Create multiple df grouped by cluster
cluster_dfs = []
for cluster_num in range(n_clusters):
    cluster_df = ts_df[ts_df["cluster"] == cluster_num]
    # sort by start_time
    cluster_df = cluster_df.sort_values("start_time")
    # Extract hour from start_time
    cluster_df["hour"] = cluster_df["start_time"].dt.floor("H")

    # Group by hour and aggregate ride_value by summation
    cluster_df = cluster_df.groupby("hour")["ride_value"].sum().reset_index()

    # Sort by time
    cluster_df = cluster_df.sort_values("hour")

    cluster_df = cluster_df.reset_index(drop=True)

    # add missing hours with average value (kind of noise)
    min_hour = cluster_df["hour"].min()
    max_hour = cluster_df["hour"].max()
    all_hours = pd.date_range(min_hour, max_hour, freq="H")
    cluster_df = cluster_df.set_index("hour")
    cluster_df = cluster_df.reindex(
        all_hours, fill_value=np.mean(cluster_df["ride_value"])
    )

    cluster_df = cluster_df.reset_index()
    cluster_df = cluster_df.rename(columns={"index": "hour"})
    # create new df with hour and ride_value
    # keep only hour and ride_value columns
    cluster_df = cluster_df[["hour", "ride_value"]]
    # cluster_df = pd.DataFrame(
    #     {"hour": cluster_df["hour"], "ride_value": cluster_df["ride_value"]}
    # )

    assert len(cluster_df) == len(all_hours)
    cluster_dfs.append(cluster_df)


# In[325]:


cluster_dfs[0]


# ## Train ride-value forecasting model for each cluster

# In[367]:


# from prophet import Prophet
import warnings
from collections import defaultdict
from itertools import product

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA

warnings.simplefilter("ignore", ConvergenceWarning)

# Define parameters for cross-validation
HOURS_IN_DAY = 24
DAYS_FOR_INITIAL_TRAINING = 14  # Days of training data
DAYS_FOR_VALIDATION = 3  # Validate on 3 day at a time
TEST_DAYS = 3  # Days for final testing


# In[373]:


import json
import os
import pickle


def save_model(model, filepath, overwrite=True):
    # Check if file exists and overwrite is False
    if os.path.exists(filepath) and not overwrite:
        print(f"File {filepath} already exists. Set overwrite=True to replace it.")

    # Create directory if it doesn't exist
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    try:
        with open(filepath, "wb") as f:
            pickle.dump(model, f)
        print(f"Model successfully saved to {filepath}")

    except Exception as e:
        print(f"Error saving model: {str(e)}")


def save_train_last_date(results, dir_pth):
    # Save train_last_date for each cluster into json
    train_last_dates = {
        i: res["base_model"]["train_last_date"].strftime("%Y-%m-%d %H:%M:%S")
        for i, res in enumerate(results)
    }
    with open(f"{dir_pth}/train_last_dates.json", "w") as f:
        json.dump(train_last_dates, f)


def load_train_last_date(dir_pth):
    # Load last_train_dates json
    with open(f"{dir_pth}/train_last_dates.json", "r") as f:
        last_train_dates = json.load(f)
        cluster_train_last_dates = {
            int(k): pd.to_datetime(v) for k, v in last_train_dates.items()
        }
    return cluster_train_last_dates


# In[357]:


def arima_single_train(cluster_id, df, save_dir, p=24, q=1, d=24):
    result = {}

    # Base model (no additional features)
    y = df["ride_value"]
    test_size_hours = TEST_DAYS * HOURS_IN_DAY
    train = y.iloc[:-test_size_hours].copy()
    test = y.iloc[-test_size_hours:].copy()
    train_last_date = df["hour"].iloc[len(train) - 1]
    # train_size_idx = int(len(y) * train_size)
    # train, test = y[:train_size_idx], y[train_size_idx:]

    try:
        model = ARIMA(train, order=(p, q, d))
        model_fit = model.fit()

        save_model(model_fit, f"{save_dir}/cluster{cluster_id}_arima.pkl")

        predictions = model_fit.forecast(steps=test_size_hours)
        mae = mean_absolute_error(test, predictions)
        rmse = np.sqrt(mean_squared_error(test, predictions))
        result["base_model"] = {
            "mae": mae,
            "rmse": rmse,
            "train_last_date": train_last_date,
            "predictions": predictions,
        }
    except Exception as e:
        result["base_model"] = {"error": str(e)}

    return result


# In[ ]:


# Plot forecast results
def plot_forecast(cluster_idx, mae, rmse, ax, train_data, test_data, forecast):
    # Plot training data
    ax.plot(
        train_data.index, train_data, label="Training Data", color="blue", alpha=0.7
    )
    # Plot test data
    ax.plot(test_data.index, test_data, label="Actual Test Data", color="green")
    # Plot forecast
    ax.plot(test_data.index, forecast, label="Forecast", color="red", linestyle="--")

    # Format axis - only add legend to first plot
    ax.tick_params(axis="x", rotation=45)
    # if ax == axes.flatten()[0]:  # Only add legend to first subplot
    #     ax.legend(loc="best", fontsize=8)

    # Add vertical line to separate train and test
    ax.axvline(x=train_data.index[-1], color="black", linestyle="-", alpha=0.5)

    # Set title with metrics
    ax.set_title(f"Cluster {cluster_idx} (MAE: {mae:.3f}, RMSE: {rmse:.3f})")


def plot_forecasts(results, cluster_dfs):
    # Create the figure and axes grid
    n_clusters = len(cluster_dfs)
    n_cols = 3  # Use 4 columns
    n_rows = (n_clusters + n_cols - 1) // n_cols  # Calculate rows needed

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()  # Flatten for easier indexing

    for cluster_idx, (result, cluster_df) in enumerate(zip(results, cluster_dfs)):
        mae, rmse = result["base_model"]["mae"], result["base_model"]["rmse"]
        test_size_hours = TEST_DAYS * HOURS_IN_DAY
        y = cluster_df["ride_value"]
        train = y.iloc[:-test_size_hours].copy()
        test = y.iloc[-test_size_hours:].copy()
        plot_forecast(
            cluster_idx,
            mae,
            rmse,
            axes[cluster_idx],
            train,
            test,
            result["base_model"]["predictions"],
        )

    # Hide unused subplots
    for i in range(n_clusters, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()


# ### Cross validation

# In[368]:


def cluster_forecast_train(cluster_idx, cluster_df):
    # Prepare training/validation and test sets
    test_size_hours = TEST_DAYS * HOURS_IN_DAY
    train_val_data = cluster_df.iloc[:-test_size_hours].copy()
    # test_data = cluster_df.iloc[-test_size_hours:].copy()

    print(f"\nCV on Cluster {cluster_idx}")

    results = arima_grid_search_cv(
        data=train_val_data,
        p_values=p_values,
        d_values=d_values,
        q_values=q_values,
    )

    best_p, best_d, best_q = results["best_params"]
    print(f"\nBest ARIMA parameters: ({best_p}, {best_d}, {best_q})")
    print(f"Best validation MAE + RMSE: {results['best_score']:.4f}")
    cv_best_params[cluster_idx] = (best_p, best_d, best_q)


# Function to evaluate ARIMA model with walk-forward validation
def evaluate_arima_model(df, p, d, q):
    max_train_size = DAYS_FOR_INITIAL_TRAINING * HOURS_IN_DAY
    test_size = DAYS_FOR_VALIDATION * HOURS_IN_DAY
    tscv = TimeSeriesSplit(
        n_splits=3, max_train_size=max_train_size, test_size=test_size
    )
    mae_errs, rmse_errs = [], []
    for train_idx, val_idx in tscv.split(df):
        train_data = df.iloc[train_idx].copy()
        val_data = df.iloc[val_idx].copy()

        # Set datetime as index for ARIMA

        train_data_indexed = train_data.set_index("hour")
        train_data_indexed.index = pd.DatetimeIndex(
            train_data_indexed.index.values, freq=train_data_indexed.index.inferred_freq
        )

        # Fit ARIMA model
        model = ARIMA(train_data_indexed["ride_value"], order=(p, d, q))
        model_fit = model.fit()

        # Forecast for validation periods
        forecast = model_fit.forecast(steps=len(val_data))
        mae = mean_absolute_error(val_data["ride_value"], forecast)
        rmse = np.sqrt(mean_squared_error(val_data["ride_value"], forecast))

        mae_errs.append(mae)
        rmse_errs.append(rmse)

    return np.mean(mae_errs), np.mean(rmse_errs)


# Time series cross-validation for ARIMA
def arima_grid_search_cv(data, p_values, d_values, q_values):
    """
    Grid search for ARIMA parameters using time series cross-validation

    Returns:
    --------
    dict
        Best parameters and performance metrics
    """
    best_score = float("inf")
    best_params = None
    all_results = defaultdict(lambda: {"mae": None, "rmse": None})

    for p, d, q in product(p_values, d_values, q_values):
        mae, rmse = evaluate_arima_model(df=data, p=p, d=d, q=q)

        if mae is not None and rmse is not None:
            all_results[(p, d, q)]["mae"] = mae
            all_results[(p, d, q)]["rmse"] = rmse
            # print(f"ARIMA({p},{d},{q}) - MAE: {mae:.4f}")
            # print(f"ARIMA({p},{d},{q}) - RMSE: {rmse:.4f}")

            if mae + rmse < best_score:
                best_score = mae + rmse
                best_params = (p, d, q)

    return {
        "best_params": best_params,
        "best_score": best_score,
        "all_results": all_results,
    }


# In[369]:


p_values = [6, 12, 24]
d_values = [1]
q_values = [6, 12, 24]

cv_best_params = {}

for cluster_idx, cluster_df in enumerate(cluster_dfs):
    cluster_forecast_train(cluster_idx, cluster_df)


# In[370]:


cv_best_params


# In[371]:


dir_pth = "models/baseline_cv"

cv_results = []
for cluster_id, cluster_df in enumerate(cluster_dfs):
    cv_results.append(
        arima_single_train(cluster_id, cluster_df, dir_pth, *cv_best_params[cluster_id])
    )


# In[374]:


# save_train_last_date(cv_results, dir_pth)
cluster_train_last_dates = load_train_last_date(dir_pth)


# In[375]:


plot_forecasts(cv_results, cluster_dfs)


# ## Simulate the inference

# In[ ]:


from geopy.distance import geodesic
from scipy.special import softmax


def weight_ride_values_softmax(
    current_position, locations, expected_ride_values, temperature=1.0
):
    """
    Weight expected ride values based on distance from current position using softmax.

    Parameters:
    current_position (array): [latitude, longitude] of current position
    locations (list): List of [latitude, longitude] arrays for potential rides
    expected_ride_values (list): Expected value for each ride location
    temperature (float): Controls the "temperature" of softmax distribution
                         Lower values make distribution more peaked (more sensitive to distance)
                         Higher values make distribution more uniform

    Returns:
    weighted_values (list): Distance-weighted ride values using softmax
    """
    # Calculate distances
    distances = []
    for location in locations:
        dist = geodesic(
            (current_position[0], current_position[1]), (location[0], location[1])
        ).kilometers
        distances.append(dist)

    # Convert to numpy array
    distances = np.array(distances)
    print(distances)
    # Invert distances (smaller distances should have higher weights)
    # Adding a small epsilon to avoid division by zero
    inverse_distances = 1 / (distances + 1e-10)

    # Apply softmax to get probability distribution
    # Dividing by temperature to control the "peakiness" of distribution
    weights = softmax(inverse_distances / temperature)
    weighted_values = weights * np.array(expected_ride_values)
    return weighted_values


def forecast(curr_datetime, future_steps, cluster_idx, train_last_date, model_dir):
    # Load the model
    with open(f"{model_dir}/cluster{cluster_idx}_arima.pkl", "rb") as f:
        model = pickle.load(f)

    # Check if current datetime is after the last training datetime
    if curr_datetime <= train_last_date:
        return "Cannot forecast for past or current time"

    steps = (curr_datetime - train_last_date).days + future_steps
    # Forecast
    forecast = model.forecast(steps=steps)
    return sum(forecast[-future_steps:]) / future_steps


# In[378]:


dir_pth = "models/baseline_cv"
curr_datetime = pd.to_datetime("2022-05-01 00:00:00")
curr_lat, curr_lng = df["start_lat"].mean(), df["start_lng"].mean()
print(f"Current datetime: {curr_datetime}")
print(f"Current location: Lat {curr_lat:.6f}, Lng {curr_lng:.6f}")


# Load last_train_dates json
cluster_train_last_dates = load_train_last_date(dir_pth)

future_steps = 1
prediction = [
    forecast(
        curr_datetime,
        future_steps,
        cluster_idx,
        cluster_train_last_dates[cluster_idx],
        dir_pth,
    )
    for cluster_idx in range(n_clusters)
]
# top 5 cluster with highest prediction
top_cluster_idx = np.argsort(prediction)[-5:]


# In[380]:


wrv = weight_ride_values_softmax((curr_lat, curr_lng), centers, prediction, 1)
topk = np.argsort(wrv)[-5:]
top_loc = [centers[cluster_idx] for cluster_idx in topk]


# In[381]:


top_loc


# In[382]:


get_ipython().system('jupyter nbconvert --to script ride_hailing.ipynb')


# # Ride-Hailing Demand Prediction System Report
# 
# ## 1. Data Exploration and Solution Approach
# 
# 
# My existing code covers:
# - Data loading and basic information
# - Data cleaning (removing missing values, duplicates, outliers)
# - Exploratory data analysis (EDA) with visualizations
# - Coordinate clustering to identify key locations
# - Time series analysis of ride values by cluster
# 
# ### Key Insights:
# - The data shows clear spatial patterns in ride demand
# - Temporal patterns exist by hour of day and day of week
# - K-means clustering effectively identifies high-demand areas
# - Forecasting models can predict future demand by location
# 
# ## 2. Baseline Model Solution
# 
# ### Baseline Model Documentation
# 
# The baseline solution uses a two-stage approach:
# 1. **Spatial Clustering**: Using K-means to identify high-demand geographical zones
# 2. **Time Series Forecasting**: ARIMA models for each cluster to predict future demand
# 
# **Model Performance**:
# - Cross-validation was used to find optimal ARIMA parameters
# - Models were evaluated using MAE and RMSE metrics
# - Forecast visualizations show reasonable prediction accuracy
# 
# **Inference Workflow**:
# - The model provides actionable predictions through these steps:
#   1. Receive current driver location and timestamp
#   2. Generate demand forecasts for each cluster using ARIMA models
#   3. Weight predictions by distance using softmax function
#   4. Rank and recommend optimal locations to drivers
# 
# 
# **Strengths**:
# - Captures both spatial and temporal patterns
# - Provides location-specific predictions
# - Uses established time series methods
# - Relatively simple to implement and interpret
# 
# **Limitations**:
# - Does not incorporate external factors (weather, events)
# - Limited to historical patterns
# - Assumes demand patterns remain stable
# 
# ## 3. Model Design and Deployment
# 
# ### AWS-Based System Architecture 
# ```
# ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
# │             │     │             │     │             │
# │   Amazon    │────▶│  Amazon     │────▶│     API     │
# │     S3      │     │  SageMaker  │     │   Gateway   │
# │             │     │             │     │             │
# └─────────────┘     └─────────────┘     └─────────────┘
#        ▲                   │                   │
#        │                   │                   ▼
#        │                   │            ┌─────────────┐
# ┌─────────────┐            │            │             │
# │             │            │            │    AWS      │
# │  Ride Data  │            └───────────▶│   Lambda    │
# │  Collection │                         │             │
# └─────────────┘                         └─────────────┘
#        │
#        ▼
# ┌─────────────┐
# │             │
# │ CloudWatch  │
# │ Monitoring  │
# │             │
# └─────────────┘
# ```
# 
# **1. Data Storage (Amazon S3)**
# - All ride data is automatically saved to S3 buckets
# 
# **2. Model Training (Amazon SageMaker)**
# - Weekly scheduled training jobs process historical data
# - For each geographical cluster, an ARIMA model is trained
# - Trained models are saved back to S3
# 
# **3. Prediction Service (Lambda + API Gateway)**
# - Driver's app calls API Gateway endpoint
# - Gateway triggers Lambda function
# - Lambda loads models and returns predictions
# 
# **4. Monitoring (CloudWatch)**
# 
# - Automatically tracks all system components
# - Captures logs, metrics, and alerts
# - Example metrics monitored:
# 
#     - Prediction latency (avg, p95, p99)
#     - Request volume
#     - Error rates
#     - Model accuracy
#     - Driver response to recommendations
# 
# ## 4. Driver Communication Strategy
# 
# ### Communication Channels
# 
# 1. **In-App Map Interface**:
#    - Heat map visualization of predicted demand
#    - Color-coded areas (red = high demand, blue = low demand)
#    - Suggested routes to high-demand areas
# 
# 2. **Push Notifications**:
#    - Scheduled alerts before peak hours
#    - Real-time opportunities in nearby areas
#    - Personalized recommendations based on driver location
# 
# 3. **Driver Dashboard**:
#    - Daily/weekly demand forecasts
#    - Earnings potential by area
#    - Historical performance metrics
# 
# 
# ## 5. Experiment Design for Validation
# 
# 
# ### Test Setup
# - **What:** Testing if demand forecasts help drivers earn more and reduce customer wait times
# - **How:** Give 500 drivers the forecast tool, 500 others continue as normal
# - **When:** Run for 4 weeks
# 
# ### Safety Measures
# - Check results daily to catch problems early
# - Stop the test if wait times increase by more than 10%
# - Have a plan to quickly remove the feature if needed
# 
# ### Market Balance Checks
# - Make sure drivers don't all crowd in one area
# - Keep track of neighborhoods that might be left without service
# - Watch how prices are affected
# 
# ### Results Breakdown
# - Compare earnings before and after
# - See if wait times improve
# - Look at different situations:
#   * Rush hour vs. quiet times
#   * Downtown vs. suburbs
#   * New drivers vs. veterans
# 
# ### Getting Feedback
# - Ask drivers what they think weekly
# - Review issues with the operations team
# - Make improvements based on what we learn

# 
