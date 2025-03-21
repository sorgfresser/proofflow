import pandas as pd
import matplotlib.pyplot as plt

model_files = {
    "small":  ["smallreward42.csv",  "smallreward41.csv",  "smallreward43.csv"],
    "medium": ["mediumreward42.csv","mediumreward41.csv","mediumreward43.csv"]
}


def load_and_merge_seed_data(file_list, reward_colnames):
    """Read multiple CSVs, merge them on Round and compute mean/std across seeds.
    """
    dfs = []
    for fname, rcol in zip(file_list, reward_colnames):
        df = pd.read_csv(fname)
        df.rename(columns={"Step": "Round", rcol: "reward"}, inplace=True)
        dfs.append(df[["Round", "reward"]])

    # Use an outer join so that we keep rounds from any seed.
    df_merge = dfs[0].rename(columns={"reward": "reward_seed1"})
    for i, d in enumerate(dfs[1:], start=2):
        df_merge = df_merge.merge(d, on="Round", how="outer")
        df_merge.rename(columns={"reward": f"reward_seed{i}"}, inplace=True)

    df_merge.sort_values("Round", inplace=True)
    reward_cols = [c for c in df_merge.columns if c.startswith("reward_seed")]
    df_merge["mean_reward"] = df_merge[reward_cols].mean(axis=1, skipna=True)
    df_merge["std_reward"] = df_merge[reward_cols].std(axis=1, skipna=True)

    return df_merge


model_data = {}
window_size = 10

for model_name, files in model_files.items():
    reward_cols = {
        "small": ["smallrun42again - train/sampled_mean_reward", "smallrunseed41 - train/sampled_mean_reward", "still-oath-232 - train/sampled_mean_reward"],
        "medium": ["Main - train/sampled_mean_reward", "mainseed41 - train/sampled_mean_reward", "mainseed43 - train/sampled_mean_reward"],
    }

    df = load_and_merge_seed_data(files, reward_cols[model_name])

    df = df[df["Round"] <= 200].copy()

    df.sort_values("Round", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["mean_reward_smooth"] = df["mean_reward"].rolling(window_size, min_periods=1).mean()
    df["std_reward_smooth"] = df["std_reward"].rolling(window_size, min_periods=1).mean()

    model_data[model_name] = df

plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.size': 14})

for model_name, df in model_data.items():
    line, = plt.plot(
        df["Round"],
        df["mean_reward_smooth"],
        label=f"{model_name}"
    )

    plt.fill_between(
        df["Round"],
        df["mean_reward_smooth"] - df["std_reward_smooth"],
        df["mean_reward_smooth"] + df["std_reward_smooth"],
        alpha=0.2
    )
    initial_value = df["mean_reward_smooth"].iloc[0]
    plt.axhline(
        y=initial_value,
        linestyle='--',
        color=line.get_color(),
        label=f"{model_name} initial"
    )

plt.xlabel("Round")
plt.ylabel("Reward")
plt.title("Mean ± Std (Rolling)")
plt.legend()
# plt.show()
plt.savefig("modelcomparison.png")
