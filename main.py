import argparse

import pandas as pd

from baseline import BaselineRecommender
from data import assign_user_folds, load_ratings
from evaluation import evaluate_model, print_summary
from user_knn import UserKNNRecommender


def main():
    parser = argparse.ArgumentParser(
        description="MovieLens"
    )

    parser.add_argument(
        "--ratings-path",
        type=str,
        default="ml-25m/ratings.csv",
        help="Path to ratings.csv",
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["baseline", "userknn", "both"],
        default="both",
        help="Algorithm to evaluate",
    )

    parser.add_argument(
        "--sample-users",
        type=int,
        default=3000,
        help=(
            "Number of users to sample. "
            "Use 0 to run on all users. "
            "User-kNN over the full MovieLens 25M dataset can be very slow."
        ),
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation",
    )

    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=30,
        help="Number of neighbors for User-kNN",
    )

    parser.add_argument(
        "--min-common-items",
        type=int,
        default=5,
        help="Minimum common movies needed to compute Pearson similarity",
    )

    parser.add_argument(
        "--max-candidates",
        type=int,
        default=1000,
        help=(
            "Maximum number of candidate users checked per prediction in User-kNN. "
            "Use 0 for no limit."
        ),
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    sample_users = None if args.sample_users == 0 else args.sample_users
    max_candidates = None if args.max_candidates == 0 else args.max_candidates

    ratings = load_ratings(
        path=args.ratings_path,
        sample_users=sample_users,
        random_state=args.random_state,
    )

    ratings = assign_user_folds(
        ratings,
        n_folds=args.folds,
        random_state=args.random_state,
    )

    all_results = []

    if args.algorithm in ["baseline", "both"]:
        baseline_results = evaluate_model(
            model_name="Baseline",
            model_factory=lambda: BaselineRecommender(),
            ratings=ratings,
            n_folds=args.folds,
        )

        all_results.append(baseline_results)

    if args.algorithm in ["userknn", "both"]:
        userknn_results = evaluate_model(
            model_name="User-kNN",
            model_factory=lambda: UserKNNRecommender(
                k_neighbors=args.k_neighbors,
                min_common_items=args.min_common_items,
                max_candidates=max_candidates,
                random_state=args.random_state,
            ),
            ratings=ratings,
            n_folds=args.folds,
        )

        all_results.append(userknn_results)

    results = pd.concat(all_results, ignore_index=True)

    results.to_csv("results.csv", index=False)

    print_summary(results)

    print()
    print("Results saved to results.csv")


if __name__ == "__main__":
    main()