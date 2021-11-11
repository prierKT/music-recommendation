#%%
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime

import pyaudio
import wave
import librosa
import librosa.display

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

from tslearn.metrics import dtw
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import xgboost as xgb
from xgboost import XGBClassifier

class RecomendMusic:
  
  def __init__(self, model_path):
    self.model = load_model(model_path)
  
  def feature_extraction(self, file_path):

    y, sr = librosa.core.load(file_path, sr=16000)
    chroma = librosa.feature.chroma_stft(y, sr=sr, hop_length=1024)
    chroma = chroma.reshape((1, ) + chroma.shape)
    centroid = librosa.feature.spectral_centroid(y, sr=sr, hop_length=1024)
    rolloff = librosa.feature.spectral_rolloff(y, sr=sr, hop_length=1024)
    harm, perc = librosa.effects.hpss(y)
    zero = librosa.feature.zero_crossing_rate(y, hop_length=1024)
    bpm, _ = librosa.beat.beat_track(y, sr=sr, hop_length=1024)
    mfccs = librosa.feature.mfcc(y, sr=sr, n_mfcc=13)
    mfccs = mfccs.reshape((1, ) + mfccs.shape)

    feature_list = []
    
    feature_list.append(centroid.mean())
    feature_list.append(centroid.var())
    feature_list.append(rolloff.mean())
    feature_list.append(rolloff.var())
    feature_list.append(harm.mean())
    feature_list.append(harm.var())
    feature_list.append(perc.mean())
    feature_list.append(perc.var())
    feature_list.append(zero.mean())
    feature_list.append(zero.var())
    feature_list.append(bpm.mean())

    feature_arr = np.array(feature_list).reshape(1, -1)
    # print(feature_arr.shape)
    
    feature_df = pd.DataFrame(feature_arr)
    # feature_df
    
    mfccs_mean = mfccs.mean(axis=2)
    mfccs_var = mfccs.var(axis=2)

    mfccs_mean_df = pd.DataFrame(mfccs_mean)
    mfccs_var_df = pd.DataFrame(mfccs_var)

    mfccs_df = pd.merge(mfccs_mean_df, mfccs_var_df,
                              left_index=True, right_index=True, how='left')

    chroma_mean = chroma.mean(axis=2)
    chroma_var = chroma.var(axis=2)

    chroma_mean_df = pd.DataFrame(chroma_mean)
    chroma_var_df = pd.DataFrame(chroma_var)

    chroma_df = pd.merge(chroma_mean_df, chroma_var_df,
                              left_index=True, right_index=True, how='left')

    mfccs_chroma = pd.merge(mfccs_df, chroma_df,
                              left_index=True, right_index=True, how='left')
    
    features_mfcc_chroma = pd.merge(feature_df, mfccs_chroma,
                                    left_index=True, right_index=True, how='left')
    
    return features_mfcc_chroma, mfccs


  def dtw_distance(self, mfccs):
    mfccs_np = np.load('mfccs_np.npy')
    
    dtw_df = pd.DataFrame(columns=['Dtw_distance'])
    dtw_list = []
    
    song1 = mfccs[0]
    dtw_values=[]
      
    for i in range(mfccs_np.shape[0]):
      song2 = mfccs_np[i]
      dtw_value = dtw(song1, song2)
      dtw_values.append(dtw_value)
      
    dtw_values = np.array(dtw_values)
    dtw_mean = dtw_values.mean()
    dtw_list.append(dtw_mean)
    
    dtw_df['Dtw_distance'] = dtw_list

    return dtw_df
  
  
  def make_input_data(self, features_mfcc_chroma, dtw_df):
    new_data = pd.merge(dtw_df, features_mfcc_chroma,
                    left_index=True, right_index=True, how='left')
    data_dis = pd.read_csv('data_distance.csv', encoding='utf8')
    data_dis.set_index('Title', drop=True, inplace=True)
    new_data.columns = data_dis.columns
    
    features = pd.concat([new_data, data_dis])
    features.tail(5)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    tsne = TSNE(n_components=3, random_state=111)
    tsned = tsne.fit_transform(scaled)

    tsne_df = pd.DataFrame(tsned, index=features.index)

    x = tsne_df.iloc[-1]
    X = np.array(x).reshape((1, 3, ))
    
    return X


  def recomending(self, X):
    
    dnn = self.model
    pred = dnn.predict(X)
    prediction = pred.argmax()
    
    for i in range(10):
      
      if prediction == i:
        cluster_df = 'cluster{}.csv'.format(i)
        
    print('Prediction: {}'.format(prediction))
        
    return pd.read_csv(cluster_df, encoding='utf8')[['Title', 'Label']].sample(n=5)
      

# %%
music = RecomendMusic('Dnn.h5')
feature_df, mfccs = music.feature_extraction('output.wav')
dis_df = music.dtw_distance(mfccs)
input_data = music.make_input_data(feature_df, dis_df)
music.recomending(input_data)

# %%
