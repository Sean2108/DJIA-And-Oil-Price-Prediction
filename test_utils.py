import pytest
from unittest.mock import MagicMock
from keras.models import Sequential
from utils import get_trend, get_moving_averages, calculate_mse


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ([], []),
        ([2], []),
        ([6, 5, 7, 3, 7, 10, 12, 1], [False, True, False, True, True, True, False]),
    ],
)
def test_get_trend(test_input, expected):
    assert get_trend(test_input) == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (([], 3), []),
        (([1, 2], 3), []),
        (([1, 2, 3], 3), [2]),
        (([1, 2, 3, 4, 5, 6, 7], 3), [2, 3, 4, 5, 6]),
    ],
)
def test_get_moving_averages(test_input, expected):
    assert get_moving_averages(*test_input) == expected


@pytest.mark.parametrize(
    "test_input,prediction,expected",
    [
        ([], [], 0),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 0),
        ([3, 4, 5, 6, 7], [1, 2, 3, 4, 5], 4),
        ([3, 0, 5, 2, 7], [1, 2, 3, 4, 5], 4),
    ],
)
def test_calculate_mse(test_input, prediction, expected):
    model = Sequential()
    model.predict = MagicMock(return_value=prediction)
    assert calculate_mse([], test_input, model) == expected

