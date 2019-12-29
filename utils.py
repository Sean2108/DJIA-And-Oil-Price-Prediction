import statistics as stats
import pandas as pd
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, SimpleRNN, Dropout
import numpy as np


def get_trend(data):
    return [data[i - 1] < data[i] for i in range(1, len(data))]


def get_moving_averages(data, window_size):
    return [
        stats.mean(data[i : i + window_size])
        for i in range(len(data) - window_size + 1)
    ]


def get_moving_window_dataframes(data, window_size):
    averages, trends = pd.DataFrame(), pd.DataFrame()
    for column in list(data):
        averages[column] = get_moving_averages(data[column], window_size)
        trends[column] = get_trend(averages[column])
    return averages, trends


def inverse_transform_single_column(scaler, y_values, num_variables):
    return scaler.inverse_transform(
        np.array([[y] + [0] * (num_variables - 1) for y in y_values])
    )[:, 0]


def plot_loss(history):
    plt.plot(history.history["loss"])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.show()


def calculate_mse(x_test, y_test, model):
    y_predict = model.predict(x_test)
    len_y_predict = len(y_predict)
    if len_y_predict == 0:
        return 0
    mse = 0
    for i in range(len_y_predict):
        mse += (y_predict[i] - y_test[i]) ** 2
    return mse / len_y_predict


def train_model(
    x_train, y_train, units, dropout, num_lstm_layers, model_type, epoch, batch_size
):
    model = Sequential(
        [
            LSTM(
                units,
                return_sequences=True,
                input_shape=(x_train.shape[1], x_train.shape[2]),
            )
            if model_type == "lstm"
            else GRU(
                units,
                return_sequences=True,
                input_shape=(x_train.shape[1], x_train.shape[2]),
            )
            if model_type == "gru"
            else SimpleRNN(
                units,
                return_sequences=True,
                input_shape=(x_train.shape[1], x_train.shape[2]),
            ),
            Dropout(dropout),
        ]
        + [
            LSTM(units, return_sequences=True)
            if model_type == "lstm"
            else GRU(units, return_sequences=True)
            if model_type == "gru"
            else SimpleRNN(units, return_sequences=True),
            Dropout(dropout),
        ]
        * (num_lstm_layers - 2)
        + [
            LSTM(units)
            if model_type == "lstm"
            else GRU(units)
            if model_type == "gru"
            else SimpleRNN(units),
            Dropout(dropout),
            Dense(1),
        ]
    )
    model.compile(optimizer="rmsprop", loss="mean_squared_error")
    history = model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size)
    return model, history


def plot_prediction(scaler, x_test, y_test, models):
    plt.plot(
        inverse_transform_single_column(scaler, y_test),
        color="blue",
        label="Real Price",
    )
    for name, model, color in models:
        y_predict = model.predict(x_test)
        plt.plot(
            inverse_transform_single_column(scaler, y_predict, x_test.shape[2]),
            color=color,
            label=name + " Predicted Price",
        )
    plt.title("Crude Oil Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
