import pandas as pd


def create_features(df):

    df = df.copy()

    # Family Size
    df["FamilySize"] = (
        df["SibSp"] +
        df["Parch"] + 1
    )

    # Alone
    df["IsAlone"] = (
        df["FamilySize"] == 1
    ).astype(int)

    # Cabin known
    df["CabinKnown"] = (
        df["Cabin"]
        .notna()
        .astype(int)
    )

    # Title extraction
    df["Title"] = (
        df["Name"]
        .str.extract(
            r" ([A-Za-z]+)\.",
            expand=False
        )
    )

    title_map = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs"
    }

    df["Title"] = (
        df["Title"]
        .replace(title_map)
    )

    rare_titles = [
        "Lady",
        "Countess",
        "Capt",
        "Col",
        "Don",
        "Dr",
        "Major",
        "Rev",
        "Sir",
        "Jonkheer",
        "Dona"
    ]

    df["Title"] = (
        df["Title"]
        .replace(
            rare_titles,
            "Rare"
        )
    )

    # Fare bins
    df["FareBin"] = pd.cut(
        df["Fare"],
        bins=[
            -1,
            7.91,
            14.45,
            31,
            1000
        ],
        labels=[
            "Low",
            "Medium",
            "High",
            "VeryHigh"
        ]
    )

    # Age bins
    df["AgeBin"] = pd.cut(
        df["Age"],
        bins=[
            0,
            12,
            18,
            35,
            60,
            100
        ],
        labels=[
            "Child",
            "Teen",
            "YoungAdult",
            "Adult",
            "Senior"
        ]
    )

    return df