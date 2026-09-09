import numpy as np

from metrics import clip_rating


class BaselineRecommender:

    def fit(self, train):

        self.global_mean = float(train["rating"].mean())

        user_means = train.groupby("userId")["rating"].mean()
        item_means = train.groupby("movieId")["rating"].mean()

        self.user_bias = (user_means - self.global_mean).to_dict()
        self.item_bias = (item_means - self.global_mean).to_dict()

        return self

    def predict_one(self, user_id, movie_id):

        prediction = self.global_mean

        prediction += self.user_bias.get(user_id, 0.0)
        prediction += self.item_bias.get(movie_id, 0.0)

        return clip_rating(prediction)

    def predict(self, test):

        predictions = []

        for row in test.itertuples(index=False):
            predictions.append(self.predict_one(row.userId, row.movieId))

        return np.array(predictions, dtype=np.float32)