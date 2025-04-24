import pandas as pd
import numpy as np
import sklearn.preprocessing as sk
from sklearn.model_selection import KFold

def target_encode_kfold(df, col_name, target_col, n_split=10, seed=42):
  df_new = df.copy()
  kf = KFold(n_splits=n_split, shuffle=True, random_state=seed)
  df_new[f'{col_name}'] = np.nan

  for train_idx, val_idx in kf.split(df):
    train_fold = df.iloc[train_idx]
    val_fold = df.iloc[val_idx]
    means = train_fold.groupby(col_name)[target_col].mean()

    df_new.iloc[val_idx, df_new.columns.get_loc(f'{col_name}')] = val_fold[col_name].map(means)

  global_mean = df[target_col].mean()
  df_new[f'{col_name}'] = df_new[f'{col_name}'].fillna(global_mean)
  return df_new

if __name__ == '__main__':
  dataset = './datasets/train.csv'
  df = pd.read_csv(dataset)

  # id - Drop
  df = df.drop(columns=['id'])

  # Podcast_Name
  df['Podcast_Name'] = sk.LabelEncoder().fit_transform(df['Podcast_Name'])
  #df = target_encode_kfold(df, col_name='Podcast_Name', target_col='Listening_Time_minutes')

  # Episode_Title
  df['Episode_Title'] = df['Episode_Title'].str.extract(r'(\d+)').astype(float)

  # Genre
  df = pd.get_dummies(df, columns=['Genre'], prefix='Genre')
  #df['Genre'] = sk.LabelEncoder().fit_transform(df.Genre)

  # Publication_Day
  df['Publication_Day'] = sk.LabelEncoder().fit_transform(df['Publication_Day'])
  #df['Publication_Day'] = df['Publication_Day'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)

  # Publication_Time
  df['Publication_Time'] = sk.LabelEncoder().fit_transform(df['Publication_Time'])

  # Episode_Sentiment
  sentiment_dict = {'Positive':1, 'Neutral':0, 'Negative':-1}
  df['Episode_Sentiment'] = df.Episode_Sentiment.map(sentiment_dict)

  pd.set_option('display.max_columns', None)  # 모든 열 출력
  pd.set_option('display.max_rows', 10)  # 최대 행 수도 필요 시 조절
  print(df.head())

  df.to_csv('./datasets/preprocessed.csv', index=False)
  print("Save Complete")