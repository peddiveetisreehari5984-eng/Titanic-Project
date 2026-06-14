from sklearn.pipeline import Pipeline

from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV
)

from sklearn.ensemble import (
    RandomForestClassifier
)

from src.preprocessing import (
    build_preprocessor
)

from src.config import (
    RANDOM_STATE
)


def train_model(X, y):

    preprocessor = (
        build_preprocessor()
    )

    pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                random_state=
                RANDOM_STATE
            )
        )
    ])

    param_grid = {

        "classifier__n_estimators":
            [200, 300, 500],

        "classifier__max_depth":
            [5, 8, 12, None],

        "classifier__min_samples_split":
            [2, 5, 10],

        "classifier__min_samples_leaf":
            [1, 2, 4]
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=
        RANDOM_STATE
    )

    search = (
        RandomizedSearchCV(

            pipeline,

            param_grid,

            n_iter=20,

            cv=cv,

            scoring="accuracy",

            random_state=
            RANDOM_STATE,

            n_jobs=-1
        )
    )

    search.fit(X, y)

    print(
        "\nBest Score:",
        search.best_score_
    )

    print(
        "\nBest Parameters:"
    )

    print(
        search.best_params_
    )

    return (
        search.best_estimator_
    )