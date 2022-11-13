from tensorflow.keras.layers import Dense, LSTM, Bidirectional, GRU, Dropout
from tensorflow.keras import optimizers, Sequential

def build_LSTM(
    output_size:int, loss:str='mse', 
    summarize:bool=False, learning_rate:float=1e-5  
    ):
    model = Sequential([
        LSTM(64, return_sequences=True),
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

def build_BiLSTM(
    output_size:int, loss:str='mse', 
    summarize:bool=False, learning_rate:float=1e-5
    ):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True)),
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
    output_size:int, loss:str='mse', 
    summarize:bool=False, learning_rate:float=1e-5
    ):
    model = Sequential([
        Bidirectional(GRU(64, return_sequences=True)),
        Dropout(0.1),
        Bidirectional(GRU(64, return_sequences=True)),
        Bidirectional(GRU(32)),
        Dense(64),
        Dense(output_size)
    ])
    if summarize:
        model.summary()
    
    adam = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=adam)
    return model