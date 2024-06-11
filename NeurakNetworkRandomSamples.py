from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy.stats import spearmanr
import numpy as np
import joblib

Folder = 'DataFiles\\'

year = 2023
start_date = f"{year}-04-01"
end_date = f"{year}-10-01"
final_dataframe = pd.read_pickle(f'{Folder}player_game_stats_{start_date}_to_{end_date}.pkl')

# Remove rows with NaN values
final_dataframe = final_dataframe.dropna()

# Define the sample sizes to try
sample_sizes = [1, 3, 5, 7]

# Define the number of random samples to choose
num_samples = 1000

# Select the relevant features and target variable
features = [
    'Hits_Per_Game_1_games', 'Hits_Per_Game_3_games', 'Hits_Per_Game_7_games', 'Hits_Per_Game_All_games',
    'Hits_Per_PA_1_games', 'Hits_Per_PA_3_games', 'Hits_Per_PA_7_games', 'Hits_Per_PA_All_games',
    '1_Starter', '1_MiddleReliever', '1_EndingPitcher',
    '3_Starter', '3_MiddleReliever', '3_EndingPitcher',
    '7_Starter', '7_MiddleReliever', '7_EndingPitcher',
    'All_Starter', 'All_MiddleReliever', 'All_EndingPitcher',
    'Stadium_Hits'
]
target = 'Hits'

# Create a new DataFrame with the selected features and target
selected_dataframe = final_dataframe[features + [target]]

# Initialize a dictionary to store the evaluation metrics for each sample size
metrics_dict = {}

for sample_size in sample_sizes:
    print(f"\nSample Size: {sample_size}")

    # Initialize an empty list to store the sampled DataFrames
    sampled_dfs = []

    for _ in range(num_samples):
        # Generate random indices for a single sample
        sample_indices = np.random.randint(len(selected_dataframe), size=sample_size)

        # Use the indices to select a random sample from the DataFrame
        sample_df = selected_dataframe.iloc[sample_indices]

        # Calculate the mean of the sample and convert it to a DataFrame
        sample_mean_df = pd.DataFrame(sample_df.mean()).T

        # Append the sample mean DataFrame to the list
        sampled_dfs.append(sample_mean_df)

    # Concatenate all the sampled DataFrames into a single DataFrame
    sampled_dataframe = pd.concat(sampled_dfs, ignore_index=True)

    X = sampled_dataframe[features]
    y = sampled_dataframe[target]

    # Create a scaler for all features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Create a neural network regressor
    nn_model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000, random_state=42)

    # Train the neural network
    nn_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = nn_model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    spearman_corr, _ = spearmanr(y_pred, y_test)

    # Store the evaluation metrics for the current sample size
    metrics_dict[sample_size] = {
        'MAE': mae,
        'MAPE': mape,
        'Spearman Correlation': spearman_corr
    }

    print(f"\nMean Absolute Error: {mae:.4f}")
    print(f"Mean Absolute Percentage Error: {mape:.4f}")
    print(f"Spearman Correlation: {spearman_corr:.4f}")


# Find the best sample size based on the evaluation metrics
best_sample_size = None
best_metrics = None
best_model = None

for sample_size, metrics in metrics_dict.items():
    if best_metrics is None or metrics['Spearman Correlation'] > best_metrics['Spearman Correlation']:
        best_sample_size = sample_size
        best_metrics = metrics
        best_model = nn_model  # Save the best model

print(f"\nBest Sample Size: {best_sample_size}")
print("Best Metrics:")
for metric, value in best_metrics.items():
    print(f"{metric}: {value:.4f}")

# Save the best model to a file
joblib.dump(best_model, 'best_nn_model.pkl')
