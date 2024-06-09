from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
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

#Scale stadium hits
final_dataframe['Stadium_Hits'] = final_dataframe['Stadium_Hits']/50

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
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Print the predictions and actual values side by side
print("Predictions\tActual")
for pred, actual in zip(y_pred, y_test):
    print(f"{pred:.2f}\t\t{actual}")

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