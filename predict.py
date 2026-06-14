import pandas as pd


def create_submission(
    model,
    test_df,
    output_path
):

    passenger_ids = (
        test_df["PassengerId"]
    )

    features = test_df.drop(
        columns=[
            "PassengerId",
            "Name",
            "Ticket",
            "Cabin"
        ]
    )

    predictions = (
        model.predict(features)
    )

    submission = (
        pd.DataFrame({

            "PassengerId":
                passenger_ids,

            "Survived":
                predictions
        })
    )

    submission.to_csv(
        output_path,
        index=False
    )

    print(
        "Submission saved."
    )