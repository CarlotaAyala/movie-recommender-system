import math
from collections import defaultdict

import numpy as np

from baseline import BaselineRecommender
from metrics import clip_rating


class UserKNNRecommender:

    def __init__(
        self,
        k_neighbors=30,
        min_common_items=5,
        max_candidates=1000,
        random_state=42,
    ):
        self.k_neighbors = k_neighbors
        self.min_common_items = min_common_items
        self.max_candidates = max_candidates
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def fit(self, train):

        self.baseline = BaselineRecommender().fit(train)

        self.user_means = train.groupby("userId")["rating"].mean().to_dict()

        self.user_ratings = {}
        self.item_users = defaultdict(list)

        for user_id, group in train.groupby("userId"):
            ratings_dict = dict(zip(group["movieId"].values, group["rating"].values))
            self.user_ratings[user_id] = ratings_dict

            for movie_id in ratings_dict:
                self.item_users[movie_id].append(user_id)

        self.similarity_cache = {}

        return self

    def pearson_similarity(self, user_a, user_b):
        if user_a == user_b:
            return 1.0

        key = (min(user_a, user_b), max(user_a, user_b))

        if key in self.similarity_cache:
            return self.similarity_cache[key]

        ratings_a = self.user_ratings.get(user_a)
        ratings_b = self.user_ratings.get(user_b)

        if ratings_a is None or ratings_b is None:
            self.similarity_cache[key] = 0.0
            return 0.0

        common_items = set(ratings_a.keys()).intersection(ratings_b.keys())

        if len(common_items) < self.min_common_items:
            self.similarity_cache[key] = 0.0
            return 0.0

        values_a = np.array([ratings_a[item] for item in common_items], dtype=np.float32)
        values_b = np.array([ratings_b[item] for item in common_items], dtype=np.float32)

        centered_a = values_a - values_a.mean()
        centered_b = values_b - values_b.mean()

        numerator = float(np.sum(centered_a * centered_b))
        denominator = math.sqrt(float(np.sum(centered_a ** 2))) * math.sqrt(
            float(np.sum(centered_b ** 2))
        )

        if denominator == 0.0:
            similarity = 0.0
        else:
            similarity = numerator / denominator

        self.similarity_cache[key] = similarity

        return similarity

    def predict_one(self, user_id, movie_id):

        if user_id not in self.user_ratings:
            return self.baseline.predict_one(user_id, movie_id)

        candidate_users = self.item_users.get(movie_id, [])

        if not candidate_users:
            return self.baseline.predict_one(user_id, movie_id)

        candidate_users = [user for user in candidate_users if user != user_id]

        if not candidate_users:
            return self.baseline.predict_one(user_id, movie_id)

        if self.max_candidates is not None and len(candidate_users) > self.max_candidates:
            candidate_users = self.rng.choice(
                candidate_users,
                size=self.max_candidates,
                replace=False,
            )

        neighbors = []

        for other_user in candidate_users:
            similarity = self.pearson_similarity(user_id, other_user)

            if similarity > 0:
                neighbors.append((other_user, similarity))

        if not neighbors:
            return self.baseline.predict_one(user_id, movie_id)

        neighbors.sort(key=lambda pair: pair[1], reverse=True)
        neighbors = neighbors[: self.k_neighbors]

        user_mean = self.user_means.get(user_id, self.baseline.global_mean)

        numerator = 0.0
        denominator = 0.0

        for other_user, similarity in neighbors:
            other_rating = self.user_ratings[other_user][movie_id]
            other_mean = self.user_means.get(other_user, self.baseline.global_mean)

            numerator += similarity * (other_rating - other_mean)
            denominator += abs(similarity)

        if denominator == 0.0:
            return self.baseline.predict_one(user_id, movie_id)

        prediction = user_mean + numerator / denominator

        return clip_rating(prediction)

    def predict(self, test):
        """
        Predicts all ratings in the test set.
        """
        predictions = []

        for row in test.itertuples(index=False):
            predictions.append(self.predict_one(row.userId, row.movieId))

        return np.array(predictions, dtype=np.float32)