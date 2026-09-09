import numpy as np
import pandas as pd


def load_ratings(path, sample_users=None, random_state=42):

    print("Loading ratings...")

    ratings = pd.read_csv(
        path,
        usecols=["userId", "movieId", "rating"],
        dtype={
            "userId": "int32",
            "movieId": "int32",
            "rating": "float32",
        },
    )

    if sample_users is not None:
        rng = np.random.default_rng(random_state)
        users = ratings["userId"].unique()

        if sample_users < len(users):
            selected_users = rng.choice(users, size=sample_users, replace=False)
            ratings = ratings[ratings["userId"].isin(selected_users)].copy()

    ratings = ratings.reset_index(drop=True)

    print(f"Ratings loaded: {len(ratings):,}")
    print(f"Users: {ratings['userId'].nunique():,}")
    print(f"Movies: {ratings['movieId'].nunique():,}")

    return ratings


def assign_user_folds(ratings, n_folds=5, random_state=42):

    rng = np.random.default_rng(random_state)
    fold_ids = np.empty(len(ratings), dtype=np.int8)

    for _, user_indices in ratings.groupby("userId").indices.items():
        indices = np.array(user_indices)
        rng.shuffle(indices)

        parts = np.array_split(indices, n_folds)

        for fold, part in enumerate(parts):
            fold_ids[part] = fold

    ratings = ratings.copy()
    ratings["fold"] = fold_ids

    return ratings