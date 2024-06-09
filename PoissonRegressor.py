from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy.stats import spearmanr

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
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Train the best model on the entire training set
best_model.fit(X_train, y_train)

# Print the feature weights
print("\nFeature Weights:")
for feature, weight in zip(features, best_model.coef_):
    print(f"{feature}: {weight:.4f}")

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Print the predictions and actual values side by side
# print("\nPredictions\tActual")
# for pred, actual in zip(y_pred, y_test):
#     print(f"{pred:.2f}\t\t{actual}")

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
spearman_corr, _ = spearmanr(y_pred, y_test)
print(f"\nMean Absolute Error: {mae:.4f}")
print(f"Mean Absolute Percentage Error: {mape:.4f}")
print(f"Spearman Correlation: {spearman_corr:.4f}")