import os
import librosa
from tqdm import tqdm



def song_to_data(dir_path, sampling_rate=22050):
  """노래를 음원데이터로 변환하여 Dict로 반환"""
  song_list = os.listdir(dir_path)
  
  song_data = dict()
  for song in tqdm(song_list):
    song_path = os.path.join(dir_path, song)
    
    y, sampling_rate = librosa.core.load(song_path, sr=sampling_rate)
    song_data[song.split('.')[0].replace(' ', '-')] = y[:sampling_rate*120] # 곡의 시작부터 2분 지점까지 자르기
    
  return song_data