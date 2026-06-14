from sklearn.compose import (
    ColumnTransformer
)

from sklearn.pipeline import Pipeline

from sklearn.impute import (
    SimpleImputer
)

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)


def build_preprocessor():

    numeric_features = [
        "Age",
        "Fare",
        "SibSp",
        "Parch",
        "FamilySize"
    ]

    categorical_features = [
        "Sex",
        "Embarked",
        "Pclass",
        "Title",
        "FareBin",
        "AgeBin",
        "IsAlone",
        "CabinKnown"
    ]

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ])

    return ColumnTransformer([
        (
            "num",
            numeric_pipeline,
            numeric_features
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ])