import os
import librosa
import numpy as np
from numpy import lib
import pandas as pd
import matplotlib.pyplot as plt

from sqlalchemy import create_engine

from sklearn.preprocessing import minmax_scale

from utils.preprocessing import features_mean_var



engine = create_engine("mysql+mysqldb://id:password@ip_address:port/db_name", encoding='utf-8')
conn = engine.connect()
test = pd.read_sql_table(table_name='train', con=conn, index_col='index')

data = pd.DataFrame()

sr = 4000
for name in test.columns:
  y = test[name].to_numpy()
  feature_mean_var = features_mean_var(y=y, sampling_rate=sr)
  
  for key, val in feature_mean_var.items():
    data.loc[name, key] = val