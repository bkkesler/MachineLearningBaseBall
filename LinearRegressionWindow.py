from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import nnls

Folder = 'DataFiles\\'

year = 2023
start_date = f"{year}-04-01"
end_date = f"{year}-10-01"
final_dataframe = pd.read_pickle(f'{Folder}player_game_stats_{start_date}_to_{end_date}.pkl')

# Remove rows with NaN values
final_dataframe = final_dataframe.dropna()

# Define the window sizes to try
window_sizes = [1, 3, 5, 7]

for window_size in window_sizes:
    print(f"\nWindow Size: {window_size}")

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

    # Apply the rolling window to the selected DataFrame
    window_dataframe = selected_dataframe.rolling(window=window_size, min_periods=1).mean()

    X = window_dataframe[features]
    y = window_dataframe[target]

    # Create a scaler for all features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Perform NNLS regression
    coefficients, _ = nnls(X_train, y_train)

    # Create a linear regression model with non-negative weights
    model = LinearRegression(positive=True)
    model.coef_ = coefficients
    model.intercept_ = 0

    # Print the feature weights
    print("Feature Weights:")
    for feature, weight in zip(features, coefficients):
        print(f"{feature}: {weight:.4f}")

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    spearman_corr, _ = spearmanr(y_pred, y_test)
    print(f"\nMean Absolute Error: {mae:.4f}")
    print(f"Mean Absolute Percentage Error: {mape:.4f}")
    print(f"Spearman Correlation: {spearman_corr:.4f}")