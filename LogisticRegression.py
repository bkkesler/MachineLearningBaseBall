from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
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

# Scale stadium hits
final_dataframe['Stadium_Hits'] = final_dataframe['Stadium_Hits'] / 50

features = [
    'Hits_Per_Game_All_games', 'Hits_Per_Game_3_games',
    'Hits_Per_PA_All_games',
    'All_Starter', 'All_MiddleReliever', 'All_EndingPitcher',
    'Stadium_Hits'
]
target = 'Hits'

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

# Convert the target variable to binary (1 if hits > 0, 0 otherwise)
final_dataframe['Hits_Binary'] = (final_dataframe[target] > 0).astype(int)

X = final_dataframe[features]
y = final_dataframe['Hits_Binary']

### Subsample the data to have equal amounts of entries with hits and without hits
# hits_df = final_dataframe[final_dataframe['Hits_Binary'] == 1]
# no_hits_df = final_dataframe[final_dataframe['Hits_Binary'] == 0]
#
# # Determine the minimum number of entries between hits and no hits
# min_entries = min(len(hits_df), len(no_hits_df))
#
# # Randomly sample an equal number of entries from both groups
# hits_sample = hits_df.sample(n=min_entries, random_state=42)
# no_hits_sample = no_hits_df.sample(n=min_entries, random_state=42)
#
# # Combine the subsampled dataframes
# subsampled_df = pd.concat([hits_sample, no_hits_sample])
#
# X = subsampled_df[features]
# y = subsampled_df['Hits_Binary']
###

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probability of getting at least one hit

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Print the feature weights
print("Feature Weights:")
for feature, weight in zip(features, model.coef_[0]):
    print(f"{feature}: {weight:.4f}")

# Test the model on the original dataframe
unique_dates = final_dataframe['Date'].unique()

# Function to evaluate the model on the original dataframe
def evaluate_model_on_dataframe(model):
    top1_hits, top5_hits, top10_hits = 0, 0, 0
    total_entries = 0

    for date in unique_dates:
        date_entries = final_dataframe[final_dataframe['Date'] == date]
        X_test = date_entries[features]
        y_test = date_entries['Hits_Binary']

        # Make predictions for the entries on the current date
        y_prob = model.predict_proba(X_test)[:, 1]

        # Combine the predicted probabilities with the actual hits
        results = pd.DataFrame({'predicted_prob': y_prob, 'actual_hits': y_test})

        # Sort the entries based on the predicted probabilities in descending order
        results = results.sort_values('predicted_prob', ascending=False)

        # Calculate the number of entries for the current date
        num_entries = len(results)

        # Check if the actual hits are greater than zero for the top 1, 5, and 10 entries
        if num_entries >= 1:
            top1_hits += int(results.iloc[0]['actual_hits'] > 0)
        if num_entries >= 2:
            top5_hits += (results.iloc[:2]['actual_hits'] > 0).sum()
        if num_entries >= 5:
            top10_hits += (results.iloc[:5]['actual_hits'] > 0).sum()

        total_entries += num_entries

    # Calculate the percentages of entries that got a hit for each case
    top1_percent = top1_hits / len(unique_dates) * 100
    top2_percent = top5_hits / (len(unique_dates) * 2) * 100
    top5_percent = top10_hits / (len(unique_dates) * 5) * 100

    # Print the results
    print(f"Logistic Regression - Top 1: {top1_percent:.2f}% of entries got a hit")
    print(f"Logistic Regression - Top 2: {top2_percent:.2f}% of entries got a hit")
    print(f"Logistic Regression - Top 5: {top5_percent:.2f}% of entries got a hit")

# Evaluate the logistic regression model
evaluate_model_on_dataframe(model)

print(y_prob)