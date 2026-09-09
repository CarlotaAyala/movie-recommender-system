import math
import numpy as np


MIN_RATING = 0.5
MAX_RATING = 5.0


def clip_rating(value):
    return float(min(MAX_RATING, max(MIN_RATING, value)))


def mae(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return math.sqrt(np.mean((y_true - y_pred) ** 2))