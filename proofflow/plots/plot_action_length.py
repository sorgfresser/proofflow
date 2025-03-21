import pandas as pd
import matplotlib.pyplot as plt

# Read CSV files for small and medium models
df_small_max = pd.read_csv('smallmaxlength.csv')
df_small_action = pd.read_csv('smallactionlength.csv')
df_medium_max = pd.read_csv('mediummaxlength.csv')
df_medium_action = pd.read_csv('mediumactionlength.csv')

x_small, y_small_max = df_small_max.iloc[:220, 0], df_small_max.iloc[:220, 1]
x_small_action, y_small_action = df_small_action.iloc[:220, 0], df_small_action.iloc[:220, 1]

y_small_action_smoothed = y_small_action.rolling(window=10, min_periods=0).mean()

x_medium, y_medium_max = df_medium_max.iloc[:220, 0], df_medium_max.iloc[:220, 1]
x_medium_action, y_medium_action = df_medium_action.iloc[:220, 0], df_medium_action.iloc[:220, 1]

y_medium_action_smoothed = y_medium_action.rolling(window=10, min_periods=0).mean()

plt.figure(figsize=(8, 6))
plt.rcParams.update({'font.size': 14})

plt.plot(x_small, y_small_max, label='Allowed max length', marker='o', linestyle='-')
plt.plot(x_small_action, y_small_action_smoothed, label='Small mean length', marker='s', linestyle='-')
plt.plot(x_medium_action, y_medium_action_smoothed, label='Medium mean length', marker='s', linestyle='--')

plt.xlabel('Rounds')
plt.ylabel('Trajectory length')
plt.title('Trajectory Lengths')
plt.legend()
plt.tight_layout()

plt.savefig('merged_actionlength.png')
plt.show()
