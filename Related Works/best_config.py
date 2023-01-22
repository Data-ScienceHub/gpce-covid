from dataclasses import dataclass

@dataclass
class LstmConfig:
    learning_rate = 0.000126
    batch_size = 128
    hidden_size = 64
    layers = 2
    dropout = 0.2

@dataclass
class BiLstmConfig:
    learning_rate = 0.000246
    batch_size = 128
    hidden_size = 128
    layers = 2
    dropout = 0.1

@dataclass
class NbeatsConfig:
    learning_rate = 5.1e-5
    batch_size = 64
    layers = 2
    dropout = 0.2

@dataclass
class NhitsConfig:
    learning_rate = 4.3e-05
    batch_size = 32
    layers = 3
    dropout = 0.1