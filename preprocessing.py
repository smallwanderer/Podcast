import pandas as pd
import numpy as np
import sklearn.preprocessing as sk
from sklearn.model_selection import KFold
import itertools

pd.set_option('display.max_columns', None)


class BasePreprocessor:
  def fit(self, df: pd.DataFrame):
    return self
  def transform(self, df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError
  def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
    self.fit(df)
    return self.transform(df)

class GroupMeanPreprocessor(BasePreprocessor):
  def __init__(self, target_col=None):
    if target_col is None:
      target_col = ['Guest_Popularity_percentage', 'Episode_Length_minutes']
    self.target_col = target_col
    self.fill_map = {}

  def groupwise_variance(self, df, target_col, max_cardinality=100, min_group_count=5):
    result = []

    for col in df.columns:
      if col == target_col or df[col].nunique() > max_cardinality:
        continue

      grouped = df.groupby(col)[target_col].agg(['mean', 'count'])
      grouped = grouped[grouped['count'] >= min_group_count]  # 최소 그룹 수 제한

      if len(grouped) > 1:
        variance = grouped['mean'].var()
        result.append((col, variance, len(grouped)))

    return pd.DataFrame(result, columns=['column', 'group_mean_variance', 'group_count']).sort_values(
      by='group_mean_variance', ascending=False)


  def groupwise_variance_combinations_with_group_stats(self, df, target_col, group_cols, min_group_count=3):
    results = []

    for r in range(1, len(group_cols) + 1):
      for combo in itertools.combinations(group_cols, r):
        combo = list(combo)
        grouped = df.groupby(combo)[target_col].agg(['mean', 'count'])

        filtered = grouped[grouped['count'] >= min_group_count]

        if len(filtered) > 1:
          group_variance = filtered['mean'].var()
          group_count = len(filtered)
          min_count = filtered['count'].min()
          mean_count = filtered['count'].mean()
          max_count = filtered['count'].max()

          results.append({
            'columns': ' + '.join(combo),
            'variance': group_variance,
            'group_count': group_count,
            'min_group_size': min_count,
            'mean_group_size': mean_count,
            'max_group_size': max_count
          })

    return pd.DataFrame(results).sort_values(by='variance', ascending=False)

  """
  *주의* 본 함수는 HARD CODING 되어 있음
  동작:
  결측치를 설명력이 높은 컬럼 조합(group)을 기반으로 계산된 평균값으로 계층적으로 순차적으로 채우되,
  학습 데이터 기준으로 계산한 평균값을 테스트 데이터에도 동일하게 재사용할 수 있도록 만드는 전처리기입니다.
  선택되는 칼럼명의 결합은 도메인 기반의 판단으로 선택되었으므로
  해당 함수의 수정이 필요하면 eda.py에서 각 컬럼의 결과값을 기준으로 수정하시길 바랍니다.
  출력:
  Groupwise Mean Imputation을 수행한 Dataframe
  """
  def fit(self, df):
    for target in self.target_col:
      variances = self.groupwise_variance(df, target)
      result = self.groupwise_variance_combinations_with_group_stats(
        df, target_col=target, group_cols=variances['column'][:3].tolist()
      )
      # Episode_Length_minutes
      if target == "Episode_Length_minutes":
        group_col = result['columns'][1:5].apply(lambda x: x.split(' + ')).tolist()
      # Guest_Popularity_percentage
      else:
        group_col = result.drop(index=1)['columns'].apply(lambda x: x.split(' + ')).tolist()

      self.fill_map[target] = []
      for group in group_col:
        means = df.groupby(group)[target].mean()
        self.fill_map[target].append((group, means))

    return self

  def transform(self, df):
    df = df.copy()

    for target in self.target_col:
      mask = df[target].isna()
      for group, means in self.fill_map[target]:
        if df[target].isna().sum() == 0:
          break
        fill_values = df.loc[mask, group].apply(lambda row: means.get(tuple(row), np.nan), axis=1)
        df.loc[mask, target] = df.loc[mask, target].fillna(fill_values)
        mask = df[target].isna()
      print(f"{target} 결측치 처리 완료 \n" if df[target].isna().sum() == 0 else f"{target} 실패")
    return df

  def fit_transform(self, df):
    self.fit(df)
    return self.transform(df)


class FillPreprocessor(BasePreprocessor):
  def __init__(self, method='mean', target=None):
    if target is None:
        target = ['Number_of_Ads']
    self.target = target
    self.method = method
    self.fill_map = {}

  def fit(self, df):
    for target in self.target:
      if self.method == 'mean':
        self.fill_map[target] = df[target].mean()
      elif self.method == 'median':
        self.fill_map[target] = df[target].median()
      elif self.method == 'mode':
        mode_series = df[target].mode()
        self.fill_map[target] = mode_series.iloc[0] if not mode_series.empty else np.nan
      else:
        raise ValueError("Invalid Fill Method")

  def transform(self, df):
    df = df.copy()
    for target in self.target:
      df[target] = df[target].fillna(self.fill_map[target])
      print(f"{target} 결측치 처리 완료 \n" if df[target].isna().sum() == 0 else f"{target} 실패")
    return df

  def fit_transform(self, df):
    self.fit(df)
    return self.transform(df)

class BaseEncoder:
  def fit(self, df: pd.DataFrame):
    return self
  def transform(self, df: pd.DataFrame) -> pd.DataFrame:
    return NotImplementedError
  def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
    self.fit(df)
    return self.transform(df)

class LabelEncoding(BaseEncoder):
  def __init__(self, nominal_col=None):
    if nominal_col is None:
      nominal_col = ['Podcast_Name', 'Publication_Day', 'Publication_Time']
    self.nominal_col = nominal_col
    self.encoder = {}

  def fit(self, df):
    for col in self.nominal_col:
      le = sk.LabelEncoder()
      le.fit(df[col])
      self.encoder[col] = le
    return self

  def transform(self, df):
    df = df.copy()
    for col in self.nominal_col:
      df[col] = self.encoder[col].transform(df[col])
      print(f"Label Encoding {col} encoded\n")
    return df


class OneHotEncoding(BaseEncoder):
  def __init__(self, columns):
    self.columns = columns
    self.drop_column = None

  def fit(self, df):
    dummy_df = pd.get_dummies(df[self.columns], prefix=self.columns, prefix_sep="_")
    dummy_columns = [
      col for col in dummy_df.columns
      if any(col.startswith(f"{feature}_") for feature in self.columns)
    ]

    dummy_columns = sorted(dummy_columns)
    self.drop_column = dummy_columns[0] if dummy_columns else None

  def transform(self, df):
    df = df.copy()

    dummies = pd.get_dummies(df[self.columns], prefix=self.columns, prefix_sep="_", drop_first=True)

    if self.drop_column in dummies:
      dummies = dummies.drop(columns=[self.drop_column])
      print(f"One Hot Removed column: {self.drop_column}")
    print(f"One Hot Generated columns : {len(dummies.columns.tolist())} \n")

    df = df.drop(columns=self.columns)
    df = pd.concat([df, dummies], axis=1)
    return df


class TargetEncodingKFold(BaseEncoder):
  """
  K-Fold 기반 Target Encoding 함수

  이 함수는 주어진 범주형 변수(col_name)에 대해 target 값(target_col)의 평균을 이용하여 인코딩을 수행합니다.
  단, 전체 데이터셋에 대해 단순 groupby 평균을 계산하는 방식은 데이터 누수(leakage)를 발생시킬 수 있으므로,
  이 함수는 KFold 방식으로 훈련-검증을 나누어 검증 fold에는 훈련 fold의 평균값만 반영되도록 설계되어 있습니다.

  Parameters:
      df (pd.DataFrame): 인코딩할 전체 데이터셋
      col_name (str): 인코딩할 범주형 변수명
      target_col (str): 예측 대상이 되는 target 변수명
      n_split (int): KFold의 분할 수 (기본값 10)
      seed (int): 랜덤 시드 (기본값 42)

  Returns:
    col_name_te (pd.DataFrame) : 타깃 인코딩 된 칼럼이 포함된 DataFrame
      train : KFold를 이용한 인코딩 방식을 사용
      test : fit()에서 저장된 col_name의 groupby() 평균값을 사용
  """
  def __init__(self, col_name, target_col, n_split=10, seed=42):
    self.col_name = col_name
    self.target_col = target_col
    self.n_split = n_split
    self.seed = seed
    self.global_mean = None
    self.category_mean_map = None

  def fit(self, df):
    # 전체 평균 저장
    self.global_mean = df[self.target_col].mean()

    # 전체 train set 기준 평균값 저장 (test용)
    self.category_mean_map = df.groupby(self.col_name)[self.target_col].mean()
    return self

  def transform(self, df):
    df_new = df.copy()
    col_te = self.col_name + "_te"

    if self.target_col in df.columns:
      df_new[col_te] = np.nan
      kf = KFold(n_splits=self.n_split, shuffle=True, random_state=self.seed)

      for train_idx, val_idx in kf.split(df):
        train_fold = df.iloc[train_idx]
        val_fold = df.iloc[val_idx]
        means = train_fold.groupby(self.col_name)[self.target_col].mean()

        df_new.iloc[val_idx, df_new.columns.get_loc(col_te)] = val_fold[self.col_name].map(means)

      df_new[col_te] = df_new[col_te].fillna(self.global_mean)

    else:
      df_new[col_te] = df_new[self.col_name].map(self.category_mean_map)
      df_new[col_te] = df_new[col_te].fillna(self.global_mean)

    return df_new.drop(columns=[self.col_name])

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

class DictMapping(BaseEncoder):
  def __init__(self, column, mapping_dict):
    self.column = column
    self.mapping_dict = mapping_dict

  def fit(self, df):
    return self

  def transform(self, df):
    df = df.copy()
    df[self.column] = df[self.column].map(self.mapping_dict)
    return df