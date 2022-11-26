from tensorflow.keras.layers import Dense, LSTM, Bidirectional, GRU, Dropout, Conv1D

from tensorflow.keras.layers import MultiHeadAttention, GlobalAveragePooling1D, LayerNormalization
from tensorflow.keras import optimizers, Sequential, Input, Model
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

def transformer_encoder(
    inputs, head_size, num_heads, ff_dim, 
    dropout=0, activation='relu'
    ):
    # Attention and Normalization
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = Conv1D(filters=ff_dim, kernel_size=1, activation=activation)(res)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    return x + res

def build_model(
    input_shape: Tuple[int],
    output_size:int,
    head_size:int=64,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=2,
    mlp_units=[128],
    activation='relu',
    dropout=0,
    mlp_dropout=0,
):
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(
            x, head_size, num_heads, 
            ff_dim, dropout, activation
        )

    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = Dense(dim, activation=activation)(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(output_size)(x)
    return Model(inputs, outputs)