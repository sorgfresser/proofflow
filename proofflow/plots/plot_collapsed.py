import pandas as pd
import matplotlib.pyplot as plt

df_batch1 = pd.read_csv("collapsesmallbatch1.csv")
df_batch4 = pd.read_csv("collapsesmallbatch4.csv")
df_batch4 = df_batch4.iloc[:750]

plt.figure(figsize=(8, 3))
plt.rcParams.update({'font.size': 14})
plt.plot(df_batch1['Step'],
         df_batch1['toasty-sweep-1 - train/sampled_mean_reward'],
         label='Batch size 1')
plt.plot(df_batch4['Step'],
         df_batch4['sweet-sweep-3 - train/sampled_mean_reward'],
         label='Batch size 4')

plt.xlabel('Round')
plt.ylabel('Mean Reward')
plt.title('Model Collapse')
plt.legend()

# plt.show()
plt.savefig("collapsed.png")
