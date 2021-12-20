import pandas as pd
from termcolor import colored

from sqlalchemy import create_engine

from utils.convert_data import song_to_data

sr = 4000
datasets = ['train', 'test']

for dataset in datasets:
  song_data = song_to_data(
                          dir_path=f'E:\\MusicRecommend\\dataset\\{dataset}',
                          sampling_rate=sr)

  df = pd.DataFrame()
  for key, value in song_data.items():
    df[key] = value
  print(df.shape)

  engine = create_engine("mysql+mysqldb://root:wntlr14#@34.146.75.72:3306/MusicData", encoding='utf-8')
  conn = engine.connect()
  df.to_sql(name=f'{dataset}', con=engine, if_exists='append', index=True)
  print(colored(f"{dataset} DB 입력 완료.", "cyan"))