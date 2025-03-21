import pandas as pd
import matplotlib.pyplot as plt


def plot_accuracy_vs_epoch(
        csv_small_acc, csv_small_epoch,
        csv_medium_acc, csv_medium_epoch,
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
        on='Step', how='inner'
    )

    df_medium_acc = pd.read_csv(csv_medium_acc)
    df_medium_acc.rename(
        columns={'supervisedlargea100 - validation/accuracy': 'val_acc_medium'},
        inplace=True
    )

    df_medium_epoch = pd.read_csv(csv_medium_epoch)
    df_medium_epoch.rename(
        columns={'supervisedlargea100 - epoch': 'epoch_medium'},
        inplace=True
    )
    df_medium_merged = pd.merge(
        df_medium_acc[['Step', 'val_acc_medium']],
        df_medium_epoch[['Step', 'epoch_medium']],
        on='Step', how='inner'
    )
    df_large_acc = pd.read_csv(csv_large_acc)
    df_large_acc.rename(
        columns={'biggestsupervised - validation/accuracy': 'val_acc_large'},
        inplace=True
    )
    df_large_epoch = pd.read_csv(csv_large_epoch)
    df_large_epoch.rename(
        columns={'biggestsupervised - epoch': 'epoch_large'},
        inplace=True
    )
    df_large_merged = pd.merge(
        df_large_acc[['Step', 'val_acc_large']],
        df_large_epoch[['Step', 'epoch_large']],
        on='Step', how='inner'
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        df_small_merged['epoch_small'],
        df_small_merged['val_acc_small'],
        label='Small Run', marker='o'
    )

    plt.plot(
        df_medium_merged['epoch_medium'],
        df_medium_merged['val_acc_medium'],
        label='Medium Run', marker='o'
    )

    plt.plot(
        df_large_merged['epoch_large'],
        df_large_merged['val_acc_large'],
        label='Large Run', marker='o'
    )

    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy per Epoch')
    plt.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig("supervisedcompareacc.png")


if __name__ == "__main__":
    # The steps are off, so we need epoch here
    plot_accuracy_vs_epoch(
        csv_small_acc='smallsupervised.csv',
        csv_small_epoch='smallsupervisedepochs.csv',
        csv_medium_acc='mediumsupervised.csv',
        csv_medium_epoch='mediumsupervisedepochs.csv',
        csv_large_acc='largesupervised.csv',
        csv_large_epoch='largesupervisedepochs.csv'
    )
