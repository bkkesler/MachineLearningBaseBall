import pandas as pd
data = {
    'Pitcher': ['A', 'A', 'A', 'B', 'B', 'C'],
    'DateTime': ['2023-06-01', '2023-06-01', '2023-06-02', '2023-06-01', '2023-06-02', '2023-06-02'],
    'inning': [1, 2, 1, 2, 3, 1]
}
df = pd.DataFrame(data)
events_counts = df['Pitcher'].value_counts()
# hits = events_counts[['A', 'B']].sum()
hits = sum(events_counts.get(pitcher, 0) for pitcher in [ 'D'])
print(hits)
