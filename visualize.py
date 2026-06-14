import matplotlib.pyplot as plt
import seaborn as sns


def create_visualizations(df):

    plt.figure(
        figsize=(6,4)
    )

    sns.countplot(
        x="Survived",
        data=df
    )

    plt.savefig(
        "outputs/survival_distribution.png"
    )

    plt.close()

    plt.figure(
        figsize=(6,4)
    )

    sns.countplot(
        x="Sex",
        hue="Survived",
        data=df
    )

    plt.savefig(
        "outputs/gender_survival.png"
    )

    plt.close()

    plt.figure(
        figsize=(6,4)
    )

    sns.countplot(
        x="Pclass",
        hue="Survived",
        data=df
    )

    plt.savefig(
        "outputs/class_survival.png"
    )

    plt.close()