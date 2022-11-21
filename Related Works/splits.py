from dataclasses import dataclass
from pandas import to_datetime

@dataclass
class Baseline:
    train_start = to_datetime("2020-02-29")
    validation_start = to_datetime("2021-11-30")
    test_start = to_datetime("2021-12-15")
    test_end = to_datetime("2021-12-29")

@dataclass
class Split_1:
    train_start = to_datetime("2020-02-29")
    validation_start = to_datetime("2022-01-01")
    test_start = to_datetime("2022-01-16")
    test_end = to_datetime("2022-01-30")

@dataclass
class Split_2:
    train_start = to_datetime("2020-02-29")
    validation_start = to_datetime("2022-02-01")
    test_start = to_datetime("2022-02-16")
    test_end = to_datetime("2022-03-02")

@dataclass
class Split_3:
    train_start = to_datetime("2020-02-29")
    validation_start = to_datetime("2022-03-01")
    test_start = to_datetime("2022-03-16")
    test_end = to_datetime("2022-03-30")