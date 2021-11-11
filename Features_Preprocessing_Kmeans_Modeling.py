# %%
from numpy import lib
import pymysql
from sklearn import cluster
from tensorflow.python.ops.gen_array_ops import inplace_add
pymysql.install_as_MySQLdb()

from sshtunnel import SSHTunnelForwarder
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
import pandas as pd
import pymysql
from datetime import datetime
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

# %%
def query_sql(query):
  """pandas패키지를 활용하여 db에 저장되어 있는 데이터 불러오는 함수"""
  
  # DB Connection
  con = pymysql.connect(host = "127.0.0.1", port=server.local_bind_port, user='k', passwd='123a',
                        charset='utf8', autocommit=True)
  
  # start time
  start_tm = datetime.now()

  # Get a DataFrame
  global query_result
  query_result = pd.read_sql(query, con)

  # Close connection
  end_tm = datetime.now()

  # print('START TIME : ', str(start_tm))
  # print('END TIME : ', str(end_tm))
  print('ELAP time :', str(end_tm - start_tm))
  con.close()

  return query_result

# %%
mfcc_names = query_sql('SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA="mfcc_13";')
mfcc_names

# %%
chroma_names = query_sql('SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA="chroma";')
chroma_names

# %%
song_names =  pd.merge(mfcc_names, chroma_names,
                       left_on='TABLE_NAME', right_on='TABLE_NAME', how='left')
song_names

# %%
feature_df = query_sql('SELECT * FROM music.FEATURE')
feature_df

# %%
song_features =  pd.merge(song_names, feature_df,
                       left_on='TABLE_NAME', right_on='Title', how='left')
song_features

# %%
song_features = song_features.drop_duplicates('TABLE_NAME')
song_features

# %%
# songs_100 = song_features.sample(n=100) # 전체 곡 중에서 가설 테스트를 위해 100곡만 랜덤 추출
# songs_100

# %%
song_features.drop('TABLE_NAME', axis=1, inplace=True)
song_features.set_index('Title', drop=True, inplace=True)
song_features.to_csv('song_features.csv', encoding='utf8')
song_features.dropna(axis=0, inplace=True)
song_features

# %%
song_features.isnull().sum()

# %%
"""DB에 저장되어 있는 각 음원별 mfcc값, chroma값"""
data_np = []
try:
  for i in range(0, len(song_features)):
    # df = query_sql('SELECT * FROM mfcc_13.`{}`'.format(song_features.index[i]))
    df = query_sql('SELECT * FROM chroma.`{}`'.format(song_features.index[i]))
    df_t = df.T
    df_t_np = df_t.to_numpy()
    data_np.append(df_t_np)
    
    # df_mfccs = df_mfccs.append(pd.Series(df_mean, index=df_mfccs.columns), ignore_index=True)
    print('{}곡 완료'.format(i+1))
    print('--'*30, end='\n')
except Exception as e:
  print(e)

# mfccs_np = np.array(data_np)
chroma_np = np.array(data_np)

# %%
# print(mfccs_np)
# print(mfccs_np.shape)
print(chroma_np)
print(chroma_np.shape)

#%%
# np.save('mfccs_np.npy', mfccs_np)
# np.save('chroma_np.npy', chroma_np)

# %%
# mfccs_np_re = mfccs_np.reshape(len(mfccs_np), -1)
# print(mfccs_np_re.shape)

# %%
mfccs_arr = np.load('mfccs_np.npy')
print(type(mfccs_arr))
print(mfccs_arr.shape)

# %%
mfccs_means = mfccs_arr.mean(axis=2)
print(mfccs_means)
print(mfccs_means.shape)

# %%
mfccs_means_df = pd.DataFrame(mfccs_means, index=song_features.index)
mfccs_means_df

# %%
mfccs_vars = mfccs_arr.var(axis=2)
print(mfccs_vars)
print(mfccs_vars.shape)

# %%
mfccs_vars_df = pd.DataFrame(mfccs_vars, index=song_features.index)
mfccs_vars_df

# %%
mfcc_df = pd.merge(mfccs_means_df, mfccs_vars_df,
                       left_index=True, right_index=True, how='left')
mfcc_df

# %%
chroma_np = np.load('chroma_np.npy')
print(chroma_np)
print(chroma_np.shape)

# %%
chroma_means = chroma_np.mean(axis=2)
chroma_vars = chroma_np.var(axis=2)

print(chroma_means.shape)
print(chroma_vars.shape)

# %%
chroma_means_df = pd.DataFrame(chroma_means, index=song_features.index)
chroma_vars_df = pd.DataFrame(chroma_vars, index=song_features.index)

# %%
chroma_df = pd.merge(chroma_means_df, chroma_vars_df,
                          left_index=True, right_index=True, how='left')
chroma_df

# %%
mfcc_chroma_df = pd.merge(mfcc_df, chroma_df,
                          left_index=True, right_index=True, how='left')
mfcc_chroma_df

# %%
other_fetures = song_features.drop(['Chroma_mean', 'Chroma_var'], axis=1)
other_fetures

# %%
data = pd.merge(other_fetures, mfcc_chroma_df,
                left_index=True, right_index=True, how='left')
data

# %%
from tslearn.metrics import dtw

def dtw_distance(mfcc_arr):
  dtw_df = pd.DataFrame(columns=['Dtw_distance'])
  dtw_list = []
  
  for i in range(mfccs_arr.shape[0]):
    song1 = mfccs_arr[i]
    dtw_values=[]
    
    for j in range(mfccs_arr.shape[0]):
      song2 = mfccs_arr[j]
      dtw_value = dtw(song1, song2)
      dtw_values.append(dtw_value)
      
    dtw_values = np.array(dtw_values)
    dtw_mean = dtw_values.mean()
    dtw_list.append(dtw_mean)
    
  dtw_df['Dtw_distance'] = dtw_list
  
  return dtw_df

# %%
dtw_distance_df = dtw_distance(mfccs_arr)
dtw_distance_df

# %%
dtw_distance_df.index = song_features.index
dtw_distance_df

# %%
data_dis_df = pd.merge(dtw_distance_df, data,
                       left_index=True, right_index=True, how='left')
data_dis_df

# %%
# data_dis_df.to_csv('data_distance.csv', encoding='utf8', index=True)

# %%
data_dis = pd.read_csv('data_distance.csv', encoding='utf8')
data_dis

# %%
feature_mfcc = pd.read_csv('features_mfcc.csv', encoding='utf8')
feature_mfcc

# %%
data_dis.info()

# %%
data_dis.set_index('Title', drop=True, inplace=True)
data_dis

# %%
feature_mfcc.set_index('Title', drop=True, inplace=True)
feature_mfcc

# %%
from sklearn.cluster import KMeans

def elbow(df):
  """최적의 클러스터 수를 찾기 위한 Elbow기법 함수"""
  sse = []
  
  for i in range(1, 31):
    km = KMeans(n_clusters=i, init='k-means++', random_state=111)
    km.fit(df)
    sse.append(km.inertia_)
    
  plt.plot(range(1, 31), sse, marker='o')
  plt.xlabel('Num of Cluster')
  plt.ylabel('SSE')
  plt.show()

# %%
elbow(data_dis)
# elbow(feature_mfcc)

# %%
km = KMeans(n_clusters=5, random_state=111)
label = km.fit_predict(data_dis)
print(label)

# %%
X = data_dis.to_numpy()
print(X)
print(X.shape)

# %%
import mglearn

mglearn.discrete_scatter(X[:, 0], X[:, 1], label)

# %%
data_dis['Label'] = label
data_dis

# %%
data_dis.loc[data_dis.Label == 0][['Label']]

# %%
data_dis.loc[data_dis.Label == 1][['Label']]

# %%
data_dis.loc[data_dis.Label == 2][['Label']]

# %%
data_dis.loc[data_dis.Label == 3][['Label']]

# %%
data_dis.loc[data_dis.Label == 4][['Label']]

# %%
data_dis.drop('Label', axis=1, inplace=True)

# %%
from sklearn.preprocessing import MinMaxScaler
"""사이즈가 다른 특징들을 정규화 후 다시 클러스터링"""

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_dis)

scaled_df = pd.DataFrame(scaled_data, index=data_dis.index, columns=data_dis.columns)
scaled_df

# %%
elbow(scaled_df)

# %%
km = KMeans(n_clusters=15)
label = km.fit_predict(scaled_df)
scaled_df['Label'] = label
scaled_df

# %%
scaled_df.loc[scaled_df.Label == 0][['Label']]

# %%
scaled_df.loc[scaled_df.Label == 1][['Label']]

# %%
scaled_df.loc[scaled_df.Label == 2][['Label']]

# %%
scaled_df.loc[scaled_df.Label == 3][['Label']]

# %%
scaled_df.loc[scaled_df.Label == 4][['Label']]

# %%
scaled_df.loc[scaled_df.Label == 5][['Label']]

# %%
scaled_df.loc[scaled_df.Label == 6][['Label']]

# %%
scaled_df.loc[scaled_df.Label == 7][['Label']]

# %%
scaled_df.loc[scaled_df.Label == 8][['Label']]

# %%
scaled_df.loc[scaled_df.Label == 9][['Label']]

# %%
scaled_df.loc[scaled_df.Label == 10][['Label']]

# %%
scaled_df.loc[scaled_df.Label == 11][['Label']]

# %%
scaled_df.loc[scaled_df.Label == 12][['Label']]

# %%
scaled_df.loc[scaled_df.Label == 13][['Label']]

# %%
scaled_df.loc[scaled_df.Label == 14][['Label']]

# %%
from sklearn.manifold import TSNE
"""manifold학습을 통한 차원 축소 (TSNE활용)"""

tsne = TSNE(n_components=3, random_state=111)
features_tsne = tsne.fit_transform(scaled_df)

# %%
features_tsne.shape

# %%
features_tsne_df = pd.DataFrame(features_tsne, index=scaled_df.index)
features_tsne_df

# %%
elbow(features_tsne_df)

# %%
km = KMeans(n_clusters=10, random_state=111)
label = km.fit_predict(features_tsne)
features_tsne_df['Label'] = label

features_tsne_df[['Label']]
features_tsne_df

# %%
features_tsne_df[['Label']]

# %%
color=[]
for n in range(features_tsne.shape[1]):
    if n==0:
        color.append('r')
    elif n==1:
        color.append('g')
    else:
        color.append('b')
fig = plt.figure(figsize = (8,8))
ax = fig.gca(projection='3d' )
ax.scatter(features_tsne[:, 0], features_tsne[:, 1],
           features_tsne[:, 2], alpha=0.5, c=color)
ax.set_xlabel('TSNE_1')
ax.set_ylabel('TSNE_2')
ax.set_zlabel('TSNE_3')
plt.show()

# %%
features_tsne_df.loc[features_tsne_df.Label == 0]

# %%
features_tsne_df.loc[features_tsne_df.Label == 1]

# %%
features_tsne_df.loc[features_tsne_df.Label == 2]

# %%
features_tsne_df.loc[features_tsne_df.Label == 3]

# %%
features_tsne_df.loc[features_tsne_df.Label == 4]

# %%
features_tsne_df.loc[features_tsne_df.Label == 5]

# %%
features_tsne_df.loc[features_tsne_df.Label == 6]

# %%
features_tsne_df.loc[features_tsne_df.Label == 7]

# %%
features_tsne_df.loc[features_tsne_df.Label == 8]

# %%
features_tsne_df.loc[features_tsne_df.Label == 9]

# %%
features_tsne_df.loc[features_tsne_df.Label == 10]

# %%
features_tsne_df.loc[features_tsne_df.Label == 11]

# %%
features_tsne_df.loc[features_tsne_df.Label == 12]

# %%
features_tsne_df.loc[features_tsne_df.Label == 13]

# %%
features_tsne_df.loc[features_tsne_df.Label == 14]

# %%
features_tsne_df.loc[features_tsne_df.Label == 15]

# %%
features_tsne_df.loc[features_tsne_df.Label == 16]

# %%
features_tsne_df.loc[features_tsne_df.Label == 17]

# %%
features_tsne_df.loc[features_tsne_df.Label == 18]

# %%
features_tsne_df.loc[features_tsne_df.Label == 19]

# %%
features_tsne_df.loc[features_tsne_df.Label == 20]

# %%
features_tsne_df.loc[features_tsne_df.Label == 21]

# %%
features_tsne_df.loc[features_tsne_df.Label == 22]

# %%
features_tsne_df.loc[features_tsne_df.Label == 23]

# %%
features_tsne_df.loc[features_tsne_df.Label == 24]

# %%
server.stop()

# %%
X = features_tsne_df.drop('Label', axis=1)
y = features_tsne_df['Label']

# X = features_100.drop('Label', axis=1)
# y = features_100['Label']

# X = features_tsne_df.drop('Label', axis=1)
# y = features_tsne_df['Label']

# %%
from sklearn.model_selection import train_test_split, GridSearchCV

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=111)

# %%
from xgboost import XGBClassifier ## XGBoost 불러오기
from xgboost import plot_importance ## Feature Importance를 불러오기 위함
"""GridSearchCV를 통해 최적의 파라미터 값 찾기"""

xgb = XGBClassifier()

params = {'max_depth' : [3, 5, 7],
          'learning_rate' : [0.01, 0.05, 0.75, 0.1],
          'n_estimators' : [100, 500, 1000]
          }

xgb_grid = GridSearchCV(xgb, param_grid=params, n_jobs=-1, cv=5, scoring='accuracy')
xgb_grid.fit(train_x, train_y)

# %%
print('XGBoost Best Score: ', xgb_grid.best_score_)
print('XGBoost Best Params: ', xgb_grid.best_params_)

# %%
from xgboost import XGBClassifier ## XGBoost 불러오기
from xgboost import plot_importance ## Feature Importance를 불러오기 위함
from yellowbrick.classifier import ROCAUC

xgb = XGBClassifier(learning_rate=0.75, max_depth=3, n_estimators=500)

visualizer = ROCAUC(xgb, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    micro=False, macro=True, per_class=False)
visualizer.fit(train_x, train_y)
visualizer.score(test_x, test_y)
visualizer.show()

print(xgb.score(test_x, test_y))

# %%
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

from yellowbrick.classifier import ROCAUC

# Encode the non-numeric columns
X = OrdinalEncoder().fit_transform(X)
y = LabelEncoder().fit_transform(y)

# Create the train and test data
train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=42)

# Instaniate the classification model and visualizer
model = XGBClassifier(learning_rate=0.75, max_depth=3, n_estimators=500)
visualizer = ROCAUC(model, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

visualizer.fit(train_x, train_y)        # Fit the training data to the visualizer
visualizer.score(test_x, test_y)        # Evaluate the model on the test data
visualizer.show()                       # Finalize and render the figure

# %%
pred =xgb.predict(test_x)
pred

# %%
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(test_y, pred)

plt.figure(figsize=(15, 10))
sns.heatmap( cm, annot=True,
            xticklabels=['pred_0', 'pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5', 'pred_6', 'pred_7', 'pred_8', 'pred_9'],
            yticklabels=['ture_0', 'ture_1', 'ture_2', 'ture_3', 'ture_4', 'ture_5', 'ture_6', 'ture_7', 'ture_8', 'ture_9'])
plt.title('XGBoost Confusion Matrix')

plt.show()


# %%
fig, ax = plt.subplots()
plot_importance(xgb, ax=ax)

# %%
xgb_name = 'XGBoost.model'
xgb.save_model(xgb_name)

# %%
import tensorflow as tf
from keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
"""Tensorflow-GPU를 사용하여 신경망 학습"""

tf.config.experimental.list_physical_devices(device_type='GPU')

# %%
X_np = np.array(X)
X_arr = np.expand_dims(X_np, -1)
y_cat = y.astype('category')

train_x, test_x, train_y, test_y = train_test_split(X_arr, y_cat,
                                                    test_size=0.1, random_state=111)

print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)

# %%
def model_build():
  """간단한 Dense layers를 쌓은 모델"""
  model = Sequential()

  input = Input(shape=(3, ), name='input')
  output = Dense(512, activation='relu', name='hidden1')(input)
  output = Dense(256, activation='relu', name='hidden2')(output)
  output = Dense(128, activation='relu', name='hidden5')(output)
  output = Dense(64, activation='relu', name='hidden6')(output)
  output = Dense(32, activation='relu', name='hidden7')(output)
  output = Dense(25, activation='softmax', name='output')(output)

  model = Model(inputs=[input], outputs=output)

  model.compile(optimizer=Adam(lr=0.0005), loss='sparse_categorical_crossentropy',
                metrics=['acc'])
  
  return model

# %%
model = model_build()
model.summary()

# %%
history = model.fit(train_x, train_y, epochs=100, batch_size=16, validation_split=0.1)

# %%
def plot_history(history_dict):
  loss = history_dict['loss']
  val_loss = history_dict['val_loss']

  epochs = range(1, len(loss) + 1)
  fig = plt.figure(figsize=(14, 5))

  ax1 = fig.add_subplot(1, 2, 1)
  ax1.plot(epochs, loss, 'b--', label='train_loss')
  ax1.plot(epochs, val_loss, 'r:', label='validation_loss')
  ax1.set_xlabel('Epochs')
  ax1.set_ylabel('Loss')
  ax1.set_title('Loss')
  ax1.grid()
  ax1.legend()

  acc = history_dict['acc']
  val_acc = history_dict['val_acc']

  ax2 = fig.add_subplot(1, 2, 2)
  ax2.plot(epochs, acc, 'b--', label='train_accuracy')
  ax2.plot(epochs, val_acc, 'r:', label='validation_accuracy')
  ax2.set_xlabel('Epochs')
  ax2.set_ylabel('Auccracy')
  ax2.set_title('Accuracy')
  ax2.grid()
  ax2.legend()

  plt.show()

# %%
plot_history(history.history)

# %%
model.evaluate(test_x, test_y)

# %%
model.save('Dnn.h5')

# %%
cluster0 = features_tsne_df.loc[features_tsne_df.Label == 0]
cluster0.to_csv('cluster0.csv', encoding='utf8', index=True)

# %%
cluster1 = features_tsne_df.loc[features_tsne_df.Label == 1]
cluster1.to_csv('cluster1.csv', encoding='utf8', index=True)

# %%
cluster2 = features_tsne_df.loc[features_tsne_df.Label == 2]
cluster2.to_csv('cluster2.csv', encoding='utf8', index=True)

# %%
cluster3 = features_tsne_df.loc[features_tsne_df.Label == 3]
cluster3.to_csv('cluster3.csv', encoding='utf8', index=True)

# %%
cluster4 = features_tsne_df.loc[features_tsne_df.Label == 4]
cluster4.to_csv('cluster4.csv', encoding='utf8', index=True)

# %%
cluster5 = features_tsne_df.loc[features_tsne_df.Label == 5]
cluster5.to_csv('cluster5.csv', encoding='utf8', index=True)

# %%
cluster6 = features_tsne_df.loc[features_tsne_df.Label == 6]
cluster6.to_csv('cluster6.csv', encoding='utf8', index=True)

# %%
cluster7 = features_tsne_df.loc[features_tsne_df.Label == 7]
cluster7.to_csv('cluster7.csv', encoding='utf8', index=True)

# %%
cluster8 = features_tsne_df.loc[features_tsne_df.Label == 8]
cluster8.to_csv('cluster8.csv', encoding='utf8', index=True)

# %%
cluster9 = features_tsne_df.loc[features_tsne_df.Label == 9]
cluster9.to_csv('cluster9.csv', encoding='utf8', index=True)

# %%
cluster10 = features_tsne_df.loc[features_tsne_df.Label == 10]
cluster10.to_csv('cluster10.csv', encoding='utf8', index=True)

# # %%
# cluster11 = features_tsne_df.loc[features_tsne_df.Label == 11]
# cluster11.to_csv('cluster11.csv', encoding='utf8', index=True)

# # %%
# cluster12 = features_tsne_df.loc[features_tsne_df.Label == 12]
# cluster12.to_csv('cluster12.csv', encoding='utf8', index=True)

# # %%
# cluster13 = features_tsne_df.loc[features_tsne_df.Label == 13]
# cluster13.to_csv('cluster13.csv', encoding='utf8', index=True)

# # %%
# cluster14 = features_tsne_df.loc[features_tsne_df.Label == 14]
# cluster14.to_csv('cluster14.csv', encoding='utf8', index=True)

# # %%
# cluster15 = features_tsne_df.loc[features_tsne_df.Label == 15]
# cluster15.to_csv('cluster15.csv', encoding='utf8', index=True)

# # %%
# cluster16 = features_tsne_df.loc[features_tsne_df.Label == 16]
# cluster16.to_csv('cluster16.csv', encoding='utf8', index=True)

# # %%
# cluster17 = features_tsne_df.loc[features_tsne_df.Label == 17]
# cluster17.to_csv('cluster17.csv', encoding='utf8', index=True)

# # %%
# cluster18 = features_tsne_df.loc[features_tsne_df.Label == 18]
# cluster18.to_csv('cluster18.csv', encoding='utf8', index=True)

# # %%
# cluster19 = features_tsne_df.loc[features_tsne_df.Label == 19]
# cluster19.to_csv('cluster19.csv', encoding='utf8', index=True)

# # %%
# cluster20 = features_tsne_df.loc[features_tsne_df.Label == 20]
# cluster20.to_csv('cluster20.csv', encoding='utf8', index=True)

# # %%
# cluster21 = features_tsne_df.loc[features_tsne_df.Label == 21]
# cluster21.to_csv('cluster21.csv', encoding='utf8', index=True)

# # %%
# cluster22 = features_tsne_df.loc[features_tsne_df.Label == 22]
# cluster22.to_csv('cluster22.csv', encoding='utf8', index=True)

# # %%
# cluster23 = features_tsne_df.loc[features_tsne_df.Label == 23]
# cluster23.to_csv('cluster23.csv', encoding='utf8', index=True)

# # %%
# cluster24 = features_tsne_df.loc[features_tsne_df.Label == 24]
# cluster24.to_csv('cluster24.csv', encoding='utf8', index=True)

# %%
from tensorflow.python.keras.models import load_model
# tf.config.experimental.list_physical_devices(device_type='GPU')

model = load_model('Dnn.h5')

# %%
model.evaluate(test_x, test_y)

# %%
predictions = model.predict(test_x)[:10]
print(len(predictions))

# %%
for i in range(len(predictions)):
    print('Model Prediction: {}\nTrue label: {}\n'.format(np.argmax(predictions[i]), test_y[i]))
    
# %%
mfcc_names = query_sql('SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA="mfcc_13";')
mfcc_names.duplicated(['TABLE_NAME']).value_counts()

# %%
chroma_names = query_sql('SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA="chroma";')
chroma_names.duplicated(['TABLE_NAME']).value_counts()

# %%
