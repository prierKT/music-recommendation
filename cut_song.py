# %%
import os
from pathlib import Path
import pathlib
from pandas.core.frame import DataFrame
import pymysql
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

import librosa
import librosa.display
import IPython.display as ipd

# %%
files_path = 'D:\\Sangsang\\valid_song\\'
file_list = os.listdir(files_path)
print(file_list)

# %%
singer_list = []
song_list = []

for file in file_list:
  file_path = os.path.join(files_path, file)
  # print(file_path)
  
  file_name = Path(file_path).stem
  # print(file_name)
  
  singer, song = file_name.split('_')
  # print(singer)
  # print(song)
  # print('--'*15)
  
  singer = singer.replace(' ', '_')
  song = song.replace(' ', '_')
  print(singer)
  print(song)
  print('--'*15)
  
  singer_list.append(singer)
  song_list.append(song)
  
# %%
print(len(singer_list))
print(singer_list)

# %%
print(len(song_list))
print(song_list)

#%%
from pydub import AudioSegment
import math

for file in file_list:
  file_path = os.path.join(files_path, file)
  file_name = Path(file_path).stem
  
  song = AudioSegment.from_mp3(file_path)
  minute_1 = 60 * 1000
  sec_10 = 10 * 1000
  
  slice = song[minute_1:minute_1 + sec_10]
  slice.export('{}.wav'.format(file_name), format="wav")

# %%
