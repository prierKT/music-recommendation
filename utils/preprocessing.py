import librosa



def features_mean_var(y, sampling_rate):
  """음원 특성값들의 평균과 분산값을 구해 dict로 반환"""
  feature_mean_var = dict()
  
  centroid = librosa.feature.spectral_centroid(y, sr=sampling_rate)
  feature_mean_var['centroid_mean'] = centroid.mean()
  feature_mean_var['centroid_var'] = centroid.var()
  
  rolloff = librosa.feature.spectral_rolloff(y, sr=sampling_rate)
  feature_mean_var['rolloff_mean'] = rolloff.mean()
  feature_mean_var['rolloff_var'] = rolloff.var()
  
  melspec = librosa.feature.melspectrogram(y, sr=sampling_rate)
  feature_mean_var['melspec_mean'] = melspec.mean()
  feature_mean_var['melspec_var'] = melspec.var()
  
  tempo = librosa.beat.tempo(y, sr=sampling_rate)
  feature_mean_var['tempo'] = tempo
  
  bpm, _ = librosa.beat.beat_track(y, sr=sampling_rate)
  feature_mean_var['bpm'] = bpm
  
  return feature_mean_var