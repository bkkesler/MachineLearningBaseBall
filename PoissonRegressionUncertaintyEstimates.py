from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import pandas as pd
from scipy.stats import spearmanr
import numpy as np

Folder = 'DataFiles\\'

year = 2023
start_date = f"{year}-04-01"
end_date = f"{year}-10-01"
final_dataframe = pd.read_pickle(f'{Folder}player_game_stats_{start_date}_to_{end_date}.pkl')

# Remove rows with NaN values
final_dataframe = final_dataframe.dropna()

# Assuming your DataFrame is named 'final_dataframe'
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

X = final_dataframe[features]
y = final_dataframe[target]

# Create a scaler for all features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the parameter grid for grid search
param_grid = {
    'alpha': [0.0, 0.1, 0.5, 1.0],
    'max_iter': [100, 500, 1000]
}

# Create the Poisson regression model
model = PoissonRegressor()

# Perform grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Print the best hyperparameters
print("Best Hyperparameters:")
print(grid_search.best_params_)

# Perform bootstrapping to generate multiple predictions
n_bootstraps = 100
y_pred_bootstraps = []

for _ in range(n_bootstraps):
    X_train_bootstrap, y_train_bootstrap = resample(X_train, y_train, replace=True)
    best_model.fit(X_train_bootstrap, y_train_bootstrap)
    y_pred_bootstrap = best_model.predict(X_test)
    y_pred_bootstraps.append(y_pred_bootstrap)

# Calculate confidence intervals or prediction intervals
y_pred_lower = np.percentile(y_pred_bootstraps, 2.5, axis=0)
y_pred_upper = np.percentile(y_pred_bootstraps, 97.5, axis=0)
y_pred_mean = np.mean(y_pred_bootstraps, axis=0)

# Print the predictions with uncertainty estimates
print("\nPredictions\tLower CI\tUpper CI\tActual")
for pred_mean, pred_lower, pred_upper, actual in zip(y_pred_mean, y_pred_lower, y_pred_upper, y_test):
    print(f"{pred_mean:.2f}\t\t{pred_lower:.2f}\t\t{pred_upper:.2f}\t\t{actual}")

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred_mean)
mape = mean_absolute_percentage_error(y_test, y_pred_mean)
spearman_corr, _ = spearmanr(y_pred_mean, y_test)
print(f"\nMean Absolute Error: {mae:.4f}")
print(f"Mean Absolute Percentage Error: {mape:.4f}")
print(f"Spearman Correlation: {spearman_corr:.4f}")