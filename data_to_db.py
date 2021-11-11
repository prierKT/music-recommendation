# %%
import os
from pathlib import Path
import pathlib
from librosa.feature.spectral import chroma_stft, spectral_centroid
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
files_path = 'D:\\Sangsang\\songs_cut\\'
file_list = os.listdir(files_path)
print(file_list)

# %%
singer_list = []
song_list = []

for file in file_list:
  file_name = file.replace('.wav', '')
  
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

# %%
y, sr = librosa.core.load(files_path + file_list[1], sr=16000)
print(y)
print(sr)
print(y.shape)

# %%
centroid = librosa.feature.spectral_centroid(y, sr=sr, hop_length=1024)
print(centroid.shape)
rolloff = librosa.feature.spectral_rolloff(y, sr=sr, hop_length=1024)
print(rolloff.shape)
bpm, _ = librosa.beat.beat_track(y, sr=sr, hop_length=1024)
print(bpm)

# %%
harm, prec = librosa.effects.hpss(y)
print(prec)
print(prec.shape)
print(prec.mean())

# %%
prec_mel = librosa.feature.mfcc(prec, sr=sr, hop_length=1024, n_mfcc=13)
# log_prec_mel = librosa.power_to_db(prec_mel, ref=np.max)
print(prec_mel)
print(prec_mel.shape)

plt.figure(figsize=(14, 6))
librosa.display.specshow(prec_mel, sr=sr, x_axis='time', y_axis='mel')
plt.title('Mel power spectrogram')
plt.colorbar(format='%+02.0f dB')
plt.tight_layout()

# %%
import pymysql
pymysql.install_as_MySQLdb()

from sshtunnel import SSHTunnelForwarder
from sqlalchemy import create_engine
import pandas as pd

server = SSHTunnelForwarder(('3.35.134.161', 5030),
                            ssh_username="kyoungtae14",
                            ssh_password="1234",
                            remote_bind_address=('127.0.0.1', 3306))

# %%
server.start()

# %%
local_port = str(server.local_bind_port)
local_port

# %%
server

# %%
""" MFCC값 입력 """

con = pymysql.connect(host = "127.0.0.1", port=server.local_bind_port, user='k', passwd='123a',
                     db='mfcc_13', charset='utf8', autocommit=True)

cur = con.cursor()

cols = ['DATA0', 'DATA1', 'DATA2', 'DATA3', 'DATA4', 'DATA5', 'DATA6',
        'DATA7', 'DATA8', 'DATA9', 'DATA10', 'DATA11', 'DATA12']

for i in range(0, len(file_list)):
  title = file_list[i].replace('.wav', '')
  
  y, rate = librosa.core.load(files_path + file_list[i], sr=16000)
  mfccs = librosa.feature.mfcc(y, sr=rate, hop_length=1024, n_mfcc=13)
  print(mfccs.shape)
  
  df = pd.DataFrame(mfccs.T, columns=cols)
  print(df.columns)
  
  sql_1 = "CREATE TABLE `{}` (`NUMBER` FLOAT NULL DEFAULT NULL) COLLATE='utf8_unicode_ci';".format(title)
  cur.execute(sql_1)
  print("{} 테이블이 생성되었습니다.".format(title))

  for j in range(0, mfccs.shape[0]):
    sql_2 = "ALTER TABLE `{}` ADD `DATA{}` FLOAT NULL DEFAULT NULL;".format(title, j)
    cur.execute(sql_2)
    print("DATA{} 컬럼이 생성되었습니다.".format(j))
  
  sql_3 = "ALTER TABLE `{}` DROP `NUMBER`;".format(title)
  cur.execute(sql_3)
  print("NUMBER 컬럼이 삭제되었습니다.")
        
  for n in range(len(df)):
    sql_4 = "INSERT INTO `{}` VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}');".format( title ,df['DATA0'][n], df['DATA1'][n], df['DATA2'][n], df['DATA3'][n],
                                                                                                                              df['DATA4'][n], df['DATA5'][n], df['DATA6'][n], df['DATA7'][n], df['DATA8'][n],
                                                                                                                              df['DATA9'][n], df['DATA10'][n], df['DATA11'][n], df['DATA12'][n])
    cur.execute(sql_4)
    print("데이터 입력 중: ({}/{}), ({}/{})".format(i+1, len(file_list), n+1, len(df)))
    
cur.close()
con.close()

server.stop()

# %%
""" Chroma 입력 """

con = pymysql.connect(host = "127.0.0.1", port=server.local_bind_port, user='k', passwd='123a',
                     db='chroma', charset='utf8', autocommit=True)

cur = con.cursor()

cols = ['Chroma_0', 'Chroma_1', 'Chroma_2', 'Chroma_3', 'Chroma_4',
        'Chroma_5', 'Chroma_6', 'Chroma_7', 'Chroma_8', 'Chroma_9',
        'Chroma_10', 'Chroma_11']

for i in range(0, len(file_list)):
  title = file_list[i].replace('.wav', '')
  
  y, rate = librosa.core.load(files_path + file_list[i], sr=16000)
  chroma = librosa.feature.chroma_stft(y, sr=rate, hop_length=1024)
  print(chroma.shape)
  
  df = pd.DataFrame(chroma.T, columns=cols)
  print(df.columns)
  
  sql_1 = "CREATE TABLE `{}` (`NUMBER` FLOAT NULL DEFAULT NULL) COLLATE='utf8_unicode_ci';".format(title)
  cur.execute(sql_1)
  print("{} 테이블이 생성되었습니다.".format(title))

  for j in range(0, chroma.shape[0]):
    sql_2 = "ALTER TABLE `{}` ADD `DATA{}` FLOAT NULL DEFAULT NULL;".format(title, j)
    cur.execute(sql_2)
    print("DATA{} 컬럼이 생성되었습니다.".format(j))
  
  sql_3 = "ALTER TABLE `{}` DROP `NUMBER`;".format(title)
  cur.execute(sql_3)
  print("NUMBER 컬럼이 삭제되었습니다.")
        
  for n in range(len(df)):
    sql_4 = "INSERT INTO `{}` VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}');".format( title ,df['Chroma_0'][n], df['Chroma_1'][n], df['Chroma_2'][n], df['Chroma_3'][n],
                                                                                                                              df['Chroma_4'][n], df['Chroma_5'][n], df['Chroma_6'][n], df['Chroma_7'][n],
                                                                                                                              df['Chroma_8'][n], df['Chroma_9'][n], df['Chroma_10'][n], df['Chroma_11'][n])
    cur.execute(sql_4)
    print("데이터 입력 중: ({}/{}), ({}/{})".format(i+1, len(file_list), n+1, len(df)))
    
cur.close()
con.close()

server.stop()

# %%
""" Spectral_Centroid 입력 """

con = pymysql.connect(host = "127.0.0.1", port=server.local_bind_port, user='k', passwd='123a',
                     db='centroid', charset='utf8', autocommit=True)

cur = con.cursor()

cols = ['Centroid']

for i in range(0, len(file_list)):
  title = file_list[i].replace('.wav', '')
  
  y, rate = librosa.core.load(files_path + file_list[i], sr=16000)
  centroid = librosa.feature.spectral_centroid(y, sr=rate, hop_length=1024)
  print(centroid.shape)
  
  df = pd.DataFrame(centroid.T, columns=cols)
  print(df.columns)
  
  sql_1 = "CREATE TABLE `{}` (`Centroid` FLOAT NULL DEFAULT NULL) COLLATE='utf8_unicode_ci';".format(title)
  cur.execute(sql_1)
  print("{} 테이블이 생성되었습니다.".format(title))
  
  for j in range(len(df)):
    sql_2 = "INSERT INTO `{}` VALUES ('{}');".format( title ,df['Centroid'][j])
    cur.execute(sql_2)
    print("데이터 입력 중: ({}/{}), ({}/{})".format(i+1, len(file_list), j+1, len(df)))
    
cur.close()
con.close()

server.stop()

# %%
""" Spectral_Rolloff 입력 """

con = pymysql.connect(host = "127.0.0.1", port=server.local_bind_port, user='k', passwd='123a',
                     db='rolloff', charset='utf8', autocommit=True)

cur = con.cursor()

cols = ['Rolloff']

for i in range(0, len(file_list)):
  title = file_list[i].replace('.wav', '')
  
  y, rate = librosa.core.load(files_path + file_list[i], sr=16000)
  rolloff = librosa.feature.spectral_rolloff(y, sr=rate, hop_length=1024)
  print(rolloff.shape)
  
  df = pd.DataFrame(rolloff.T, columns=cols)
  print(df.columns)
  
  sql_1 = "CREATE TABLE `{}` (`Rolloff` FLOAT NULL DEFAULT NULL) COLLATE='utf8_unicode_ci';".format(title)
  cur.execute(sql_1)
  print("{} 테이블이 생성되었습니다.".format(title))
  
  for j in range(len(df)):
    sql_2 = "INSERT INTO `{}` VALUES ('{}');".format( title ,df['Rolloff'][j])
    cur.execute(sql_2)
    print("데이터 입력 중: ({}/{}), ({}/{})".format(i+1, len(file_list), j+1, len(df)))
    
cur.close()
con.close()

server.stop()

# %%
""" 특징들의 평균, 분산 입력"""

cols = ['Title', 'Chroma_mean', 'Chroma_var', 'Centroid_mean', 'Centroid_var',
        'Rolloff_mean', 'Rolloff_var', 'Harmonic_mean', 'Harmonic_var',
        'Percussive_mean', 'Percussive_var', 'Zero_crossing_mean', 'Zero_crossing_var','Bpm']
df = pd.DataFrame(columns=cols)

for i in range(0, len(file_list)):
  row = []
  
  title = file_list[i].replace('.wav', '').replace('\'', '')
  
  y, sr = librosa.core.load(files_path + file_list[i])
  chroma = librosa.feature.chroma_stft(y)
  # print('Chroma: ', chroma.mean())

  centroid = librosa.feature.spectral_centroid(y)
  # print('Centroid: ', centroid.mean())

  rolloff = librosa.feature.spectral_rolloff(y)
  # print('Rolloff: ', rolloff.mean())

  zero_cro = librosa.feature.zero_crossing_rate(y)
  # print('Zero Crossing: ', sum(zero_cro))

  bpm, _ = librosa.beat.beat_track(y)
  # print('Bpm: ', bpm)

  harm, perc = librosa.effects.hpss(y)
  # print('Harmonic: ', harm.mean())
  # print('Percussive: ', perc.mean())
  
  row.append(title)
  row.append(chroma.mean())
  row.append(chroma.var())
  row.append(centroid.mean())
  row.append(centroid.var())
  row.append(rolloff.mean())
  row.append(rolloff.var())
  row.append(harm.mean())
  row.append(harm.var())
  row.append(perc.mean())
  row.append(perc.var())
  row.append(zero_cro.mean())
  row.append(zero_cro.var())
  row.append(bpm)
  print(row)
  print()
  
  df = df.append(pd.Series(row, index=df.columns), ignore_index=True)

  print('{}/{}곡 추출완료'.format(i+1, len(file_list)))
  print('--'*30)
  print()
  print()
  
df

# %%
import time

con = pymysql.connect(host = "127.0.0.1", port=server.local_bind_port, user='k', passwd='123a',
                      db='music', charset='utf8', autocommit=True)
cur = con.cursor()

for n in range(0, len(df)):
  sql_4 = "INSERT INTO FEATURE VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(df['Title'][n], df['Chroma_mean'][n], df['Chroma_var'][n], df['Centroid_mean'][n], df['Centroid_var'][n],
                                                                                                                                    df['Rolloff_mean'][n], df['Rolloff_var'][n], df['Harmonic_mean'][n], df['Harmonic_var'][n],
                                                                                                                                    df['Percussive_mean'][n], df['Percussive_var'][n], df['Zero_crossing_mean'][n], df['Zero_crossing_var'][n],
                                                                                                                                    df['Bpm'][n])
  cur.execute(sql_4)
  print("데이터 입력 중: {}/{}".format(n+1, len(df)))
  
  time.sleep(0.3)
    
cur.close()
con.close()

#%%
server.stop()

# %%
