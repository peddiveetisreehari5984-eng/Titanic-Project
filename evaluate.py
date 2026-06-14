import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)


def evaluate_model(
    model,
    X_test,
    y_test
):

    predictions = (
        model.predict(X_test)
    )

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    print(
        "Accuracy:",
        accuracy_score(
            y_test,
            predictions
        )
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(
        figsize=(8, 6)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d"
    )

    plt.savefig(
        "outputs/confusion_matrix.png"
    )

    plt.close()