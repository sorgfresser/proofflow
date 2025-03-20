import pandas as pd
import matplotlib.pyplot as plt


def plot_mean_reward(csv_file):
    df = pd.read_csv(csv_file)
    df = df.iloc[110:280]
    # Idk wtf wanbd names are man
    df.rename(columns={
        'mainstateskip2 - train/sampled_mean_reward': 'mean_reward'
    }, inplace=True)

    # min_periods=1 to get values for the first few rows
    df['mean_reward_rolling'] = df['mean_reward'].rolling(window=10, min_periods=1).mean()

    # Plot the original and smoothed mean rewards
    plt.figure(figsize=(8, 4))
    plt.plot(df['Step'], df['mean_reward'], label='Raw Mean Reward', alpha=0.5)
    plt.plot(df['Step'], df['mean_reward_rolling'], label='10-round Rolling Average', color='red')

    plt.xlabel('Step')
    plt.ylabel('Mean Reward')
    plt.title('Mean Reward over Rounds')
    plt.legend()
    plt.tight_layout()
    plt.savefig("meanrewardsstatekip2.png")
    #plt.show()


if __name__ == "__main__":
    plot_mean_reward('mean_reward_main_stateskip2.csv')
