from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout, Conv1D, Input

from tensorflow.keras.layers import MultiHeadAttention, GlobalAveragePooling1D, LayerNormalization
from tensorflow.keras import optimizers, Sequential, Input, Model
from typing import Tuple

def build_LSTM(
        input_shape:Tuple[int], output_size:int, loss:str='mse', hidden_size:int=64,
        dropout:float=0.1, summarize:bool=True, learning_rate:float=1e-5, layers:int=3
    ):
    assert layers>0, "layers number must be positive"
    model = Sequential()
    model.add(Input(input_shape))

    if layers==1:
        model.add(LSTM(hidden_size))
    else:
        model.add(LSTM(hidden_size, return_sequences=True))

    for layer in range(1, layers):
        model.add(Dropout(dropout))
        if layer + 1 < layers:
            model.add(LSTM(hidden_size, return_sequences=True))
        else:
            model.add(LSTM(hidden_size))
    
    model.add(Dense(output_size))

    adam = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=adam)
    if summarize:
        model.summary()

    return model

def build_BiLSTM(
        input_shape:Tuple[int], output_size:int, loss:str='mse', hidden_size:int=64,
        dropout:float=0.1, summarize:bool=True, learning_rate:float=1e-5, layers:int=3
    ):
    assert layers>0, "layers number must be positive"
    model = Sequential()
    model.add(Input(input_shape))

    if layers==1:
        model.add(Bidirectional(LSTM(hidden_size)))
    else:
        model.add(
            Bidirectional(LSTM(hidden_size, return_sequences=True)) 
        )

    for layer in range(layers-1):
        model.add(Dropout(dropout))
        if layer + 1 < layers:
            model.add(Bidirectional(LSTM(hidden_size, return_sequences=True)))
        else:
            model.add(Bidirectional(LSTM(hidden_size)))
    
    model.add(Dense(output_size))
    
    adam = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=adam)
    if summarize:
        model.summary()

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

def build_transformer(
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