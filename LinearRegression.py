from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

multiple_hits_percentage = (final_dataframe['Hits'] > 0).mean() * 100
print(multiple_hits_percentage)
blah = 2/0
#Scale stadium hits
final_dataframe['Stadium_Hits'] = final_dataframe['Stadium_Hits']/50

# Assuming your DataFrame is named 'final_dataframe'
# Select the relevant features and target variable
# features = [
#     'Hits_Per_Game_1_games', 'Hits_Per_Game_3_games', 'Hits_Per_Game_7_games', 'Hits_Per_Game_All_games',
#     'Hits_Per_PA_1_games', 'Hits_Per_PA_3_games', 'Hits_Per_PA_7_games', 'Hits_Per_PA_All_games',
#     '1_Starter', '1_MiddleReliever', '1_EndingPitcher',
#     '3_Starter', '3_MiddleReliever', '3_EndingPitcher',
#     '7_Starter', '7_MiddleReliever', '7_EndingPitcher',
#     'All_Starter', 'All_MiddleReliever', 'All_EndingPitcher',
#     'Stadium_Hits'
# ]
features = [
    'Hits_Per_Game_All_games',
    'Hits_Per_PA_All_games',
    'All_Starter', 'All_MiddleReliever', 'All_EndingPitcher',
]
target = 'Hits'

X = final_dataframe[features]
y = final_dataframe[target]

# Create a scaler for all features
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Create a new model for manual weights
model2 = LinearRegression()
model2.coef_ = np.array([0.3, 0.2*4, 0.4*4, 0.1*4, 0.1*4])
model2.intercept_ = model.intercept_


# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred2 = model2.predict(X_test)

# Print the predictions and actual values side by side
# print("Predictions\tActual")
# for pred, actual in zip(y_pred, y_test):
#     print(f"{pred:.2f}\t\t{actual}")

# Evaluate the model
mse = mean_squared_error(y_test, y_pred2)
r2 = r2_score(y_test, y_pred2)
spearman_corr, _ = spearmanr(y_pred2, y_test)
print(f"\nMean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")
print(f"Spearman Correlation: {spearman_corr:.4f}")

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
spearman_corr, _ = spearmanr(y_pred, y_test)
print(f"\nMean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")
print(f"Spearman Correlation: {spearman_corr:.4f}")

# Print the feature weights
print("Feature Weights:")
for feature, weight in zip(features, model.coef_):
    print(f"{feature}: {weight:.4f}")

# Your existing code for data preprocessing, model training, and evaluation

# Test the model on the original dataframe
unique_dates = final_dataframe['Date'].unique()


# Function to evaluate a model on the original dataframe
def evaluate_model_on_dataframe(model, model_name):
    top1_hits, top5_hits, top10_hits = 0, 0, 0
    total_entries = 0

    for date in unique_dates:
        date_entries = final_dataframe[final_dataframe['Date'] == date]
        X_test = date_entries[features]
        y_test = date_entries[target]

        # Make predictions for the entries on the current date
        y_pred = model.predict(X_test)

        # Combine the predicted hits with the actual hits
        results = pd.DataFrame({'predicted_hits': y_pred, 'actual_hits': y_test})

        # Sort the entries based on the predicted hits in descending order
        results = results.sort_values('predicted_hits', ascending=False)

        # Calculate the number of entries for the current date
        num_entries = len(results)

        # Check if the actual hits are greater than zero for the top 1, 5, and 10 entries
        if num_entries >= 1:
            top1_hits += int(results.iloc[0]['actual_hits'] > 0)
        if num_entries >= 5:
            top5_hits += (results.iloc[:5]['actual_hits'] > 0).sum()
        if num_entries >= 10:
            top10_hits += (results.iloc[:10]['actual_hits'] > 0).sum()

        total_entries += num_entries

    # Calculate the percentages of entries that got a hit for each case
    top1_percent = top1_hits / len(unique_dates) * 100
    top5_percent = top5_hits / (len(unique_dates) * 5) * 100
    top10_percent = top10_hits / (len(unique_dates) * 10) * 100

    # Print the results
    print(f"{model_name} - Top 1: {top1_percent:.2f}% of entries got a hit")
    print(f"{model_name} - Top 5: {top5_percent:.2f}% of entries got a hit")
    print(f"{model_name} - Top 10: {top10_percent:.2f}% of entries got a hit")

# Evaluate both models
evaluate_model_on_dataframe(model, "Linear Regression Model")
evaluate_model_on_dataframe(model2, "Manual Weights Model")