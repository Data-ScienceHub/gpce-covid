from tensorflow.keras.layers import Dense, LSTM, Bidirectional, GRU, Dropout
from tensorflow.keras import optimizers, Sequential
from typing import Tuple


def build_LSTM(
    input_shape: Tuple[int], output_size:int, loss:str='mse', 
    summarize:bool=False, learning_rate:float=1e-5
    ):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.1),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(64),
        Dense(output_size)
    ])
    if summarize:
        model.summary()
    
    adam = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=adam)
    return model

def build_LSTM_(
    input_shape: Tuple[int], output_size:int, loss:str='mse', 
    summarize:bool=False, learning_rate:float=1e-5, return_sequences:bool=True,
    dropout:float=0, lstm_dropout:float=0, lstm_recurrent_dropout:float=0,
    lstm_nodes:int=64, dense_nodes:int=64,
    lstm_activation:str='selu', lstm_recurrent_activation:str='sigmoid'  
    ):
    model = Sequential([
        LSTM(
            lstm_nodes, input_shape=input_shape, return_sequences=return_sequences,
            dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout, 
            activation=lstm_activation, 
            recurrent_activation=lstm_recurrent_activation
        ),
        Dropout(dropout),
        LSTM(
            lstm_nodes, return_sequences=return_sequences,
            dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout, 
            activation=lstm_activation, 
            recurrent_activation=lstm_recurrent_activation
        ),
        LSTM(
            lstm_nodes, dropout=lstm_dropout, 
            recurrent_dropout=lstm_recurrent_dropout, 
            activation=lstm_activation, 
            recurrent_activation=lstm_recurrent_activation
        ),
        Dense(dense_nodes, activation=lstm_activation),
        Dense(output_size)
    ])
    if summarize:
        model.summary()
    
    adam = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=adam)
    return model

def build_BiLSTM(
    input_shape: Tuple[int], output_size:int, loss:str='mse', 
    summarize:bool=False, learning_rate:float=1e-5
    ):
    model = Sequential([
        Bidirectional(LSTM(64, input_shape=input_shape, return_sequences=True)),
        Dropout(0.1),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32)),
        Dense(64),
        Dense(output_size)
    ])
    if summarize:
        model.summary()
    
    adam = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=adam)
    return model

def build_GRU(
    input_shape: Tuple[int], output_size:int, loss:str='mse', 
    summarize:bool=False, learning_rate:float=1e-5
    ):
    model = Sequential([
        GRU(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.1),
        GRU(64, return_sequences=True),
        GRU(32),
        Dense(64),
        Dense(output_size)
    ])
    if summarize:
        model.summary()
    
    adam = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=adam)
    return model