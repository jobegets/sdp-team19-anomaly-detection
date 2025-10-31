import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest

data = pd.read_csv("../../test-data/7-12-2025/fsae-7-12 (1).csv")

features = data[["Pack Voltage","Pack Current","Pack Temp","Min Cell Voltage",
            "State of Charge","BMS Disch Lim"]]