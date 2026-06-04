import joblib

from sklearn.model_selection import (
    train_test_split
)

from src.data_loader import (
    load_data
)

from src.feature_engineering import (
    create_features
)

from src.visualize import (
    create_visualizations
)

from src.train import (
    train_model
)

from src.evaluate import (
    evaluate_model
)

from src.predict import (
    create_submission
)

from src.config import *


def main():

    print(
        "=" * 60
    )

    print(
        "TITANIC SURVIVAL PROJECT"
    )

    print(
        "=" * 60
    )

    train_df = (
        load_data(
            TRAIN_FILE
        )
    )

    create_visualizations(
        train_df
    )

    train_df = (
        create_features(
            train_df
        )
    )

    X = train_df.drop(
        columns=[
            "Survived",
            "PassengerId",
            "Name",
            "Ticket",
            "Cabin"
        ]
    )

    y = train_df[
        "Survived"
    ]

    X_train, X_test, y_train, y_test = (
        train_test_split(

            X,
            y,

            test_size=0.2,

            random_state=
            RANDOM_STATE,

            stratify=y
        )
    )

    model = train_model(
        X_train,
        y_train
    )

    evaluate_model(
        model,
        X_test,
        y_test
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    print(
        "\nModel saved."
    )

    test_df = (
        load_data(
            TEST_FILE
        )
    )

    test_df = (
        create_features(
            test_df
        )
    )

    create_submission(

        model,

        test_df,

        SUBMISSION_FILE
    )


if __name__ == "__main__":
    main()