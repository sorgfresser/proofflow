import pandas as pd
import matplotlib.pyplot as plt


def plot_accuracy_vs_epoch(
        csv_small_acc, csv_small_epoch,
        csv_large_acc, csv_large_epoch
):
    df_small_acc = pd.read_csv(csv_small_acc)
    df_small_acc.rename(
        columns={'smallsupervisedfinal - validation/accuracy': 'val_acc_small'},
        inplace=True
    )

    df_small_epoch = pd.read_csv(csv_small_epoch)
    df_small_epoch.rename(
        columns={'smallsupervisedfinal - epoch': 'epoch_small'},
        inplace=True
    )

    df_small_merged = pd.merge(
        df_small_acc[['Step', 'val_acc_small']],
        df_small_epoch[['Step', 'epoch_small']],
        on='Step',
        how='inner'
    )
    df_large_acc = pd.read_csv(csv_large_acc)
    df_large_acc.rename(
        columns={'supervisedlargea100 - validation/accuracy': 'val_acc_large'},
        inplace=True
    )
    df_large_epoch = pd.read_csv(csv_large_epoch)
    df_large_epoch.rename(
        columns={'supervisedlargea100 - epoch': 'epoch_large'},
        inplace=True
    )
    df_large_merged = pd.merge(
        df_large_acc[['Step', 'val_acc_large']],
        df_large_epoch[['Step', 'epoch_large']],
        on='Step',
        how='inner'
    )
    # Maybe some subplots as well here, I'll think about it
    plt.figure(figsize=(8, 5))
    plt.plot(
        df_small_merged['epoch_small'],
        df_small_merged['val_acc_small'],
        marker='o',
        label='Small Supervised'
    )
    plt.plot(
        df_large_merged['epoch_large'],
        df_large_merged['val_acc_large'],
        marker='o',
        label='Large Supervised'
    )
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs. Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig("supervisedcompareacc.png")


if __name__ == "__main__":
    # The steps are off, so we need epoch here
    plot_accuracy_vs_epoch(
        csv_small_acc='smallsupervised.csv',
        csv_small_epoch='smallsupervisedepochs.csv',
        csv_large_acc='mediumsupervised.csv',
        csv_large_epoch='mediumsupervisedepochs.csv'
    )
