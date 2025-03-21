import pandas as pd
import matplotlib.pyplot as plt

df_nocritic = pd.read_csv('nocriticreward.csv')
df_critic = pd.read_csv('criticreward.csv')

df_nocritic.rename(columns={'Step': 'Round', 'criticrewardcompare - train/sampled_mean_reward': 'reward'}, inplace=True)
df_critic.rename(columns={'Step': 'Round', 'Main - train/sampled_mean_reward': 'reward'}, inplace=True)

max_round_nocritic = df_nocritic["Round"].max()
max_round_critic = df_critic["Round"].max()

min_final_round = min(max_round_nocritic, max_round_critic)

df_nocritic_filtered = df_nocritic[df_nocritic["Round"] <= min_final_round].copy()
df_critic_filtered = df_critic[df_critic["Round"] <= min_final_round].copy()

df_nocritic_filtered['reward'] = df_nocritic_filtered['reward'].rolling(window=10, min_periods=1).mean()
df_critic_filtered['reward'] = df_critic_filtered['reward'].rolling(window=10, min_periods=1).mean()

plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.size': 14})
plt.plot(df_nocritic_filtered["Round"], df_nocritic_filtered["reward"],
         label="No Critic Reward", marker='o')
plt.plot(df_critic_filtered["Round"], df_critic_filtered["reward"],
         label="Critic Reward", marker='s')

plt.xlabel("Round")
plt.ylabel("Reward")
plt.title("Reward Comparison: No Critic vs Critic")
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig("criticcompare.png")