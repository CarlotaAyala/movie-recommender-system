import time

import pandas as pd

from metrics import mae, rmse


def evaluate_model(model_name, model_factory, ratings, n_folds=5):

    fold_results = []

    for fold in range(n_folds):
        print()
        print("=" * 70)
        print(f"Model: {model_name}")
        print(f"Fold {fold + 1}/{n_folds}")
        print("=" * 70)

        train = ratings[ratings["fold"] != fold].copy()
        test = ratings[ratings["fold"] == fold].copy()

        print(f"Train ratings: {len(train):,}")
        print(f"Test ratings:  {len(test):,}")

        start = time.time()

        model = model_factory()
        model.fit(train)

        y_true = test["rating"].values
        y_pred = model.predict(test)

        fold_mae = mae(y_true, y_pred)
        fold_rmse = rmse(y_true, y_pred)

        elapsed = time.time() - start

        print(f"MAE:  {fold_mae:.4f}")
        print(f"RMSE: {fold_rmse:.4f}")
        print(f"Time: {elapsed:.2f} seconds")

        fold_results.append(
            {
                "model": model_name,
                "fold": fold + 1,
                "mae": fold_mae,
                "rmse": fold_rmse,
                "time_seconds": elapsed,
            }
        )

    return pd.DataFrame(fold_results)


def print_summary(results):
    print()
    print("=" * 70)
    print("RESULTS BY FOLD")
    print("=" * 70)
    print(results.to_string(index=False))

    print()
    print("=" * 70)
    print("AVERAGE RESULTS")
    print("=" * 70)

    summary = (
        results.groupby("model")
        .agg(
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            total_time_seconds=("time_seconds", "sum"),
        )
        .reset_index()
    )

    print(summary.to_string(index=False))