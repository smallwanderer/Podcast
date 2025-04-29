import pandas as pd
import numpy as np
import sklearn.preprocessing as sk
from sklearn.model_selection import KFold
import itertools
import re

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
  해당 함수의 수정이 필요하면 vif.py에서 각 컬럼의 결과값을 기준으로 수정하시길 바랍니다.
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
    return df

  def fit_transform(self, df):
    self.fit(df)
    return self.transform(df)


class FillPreprocessor(BasePreprocessor):
  def __init__(self, method='mean', target=None):
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
    return df


class OneHotEncoding(BaseEncoder):
  def __init__(self, columns=None):
    if columns is None:
      columns = ['Genre']
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
    print(f"One Hot 인코딩 완료 다음 열 삭제 예정 {self.drop_column}...")

  def transform(self, df):
    df = df.copy()

    dummies = pd.get_dummies(df[self.columns], prefix=self.columns, prefix_sep="_", drop_first=True)

    if self.drop_column in dummies:
      dummies = dummies.drop(columns=[self.drop_column])

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
  def __init__(self, col_name, target_col, n_split=5, smoothing=10, seed=42):
    self.col_name = col_name
    self.target_col = target_col
    self.n_split = n_split
    self.smoothing = smoothing
    self.seed = seed
    self.global_mean = None
    self.category_mean_map = None

  def _compute_smooth_mean(self, group_stats, global_mean):
    """
    group_stats: DataFrame with 'mean' and 'count'
    """
    m = self.smoothing
    return (group_stats['mean'] * group_stats['count'] + global_mean * m) / (group_stats['count'] + m)

  def fit(self, df):
    self.global_mean = df[self.target_col].mean()

    stats = df.groupby(self.col_name)[self.target_col].agg(['mean', 'count'])
    smoothed = self._compute_smooth_mean(stats, self.global_mean)
    self.category_mean_map = smoothed
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

        stats = train_fold.groupby(self.col_name)[self.target_col].agg(['mean', 'count'])
        smoothed = self._compute_smooth_mean(stats, self.global_mean)

        df_new.iloc[val_idx, df_new.columns.get_loc(col_te)] = val_fold[self.col_name].map(smoothed)

      df_new[col_te] = df_new[col_te].fillna(self.global_mean)

    else:
      df_new[col_te] = df_new[self.col_name].map(self.category_mean_map)
      df_new[col_te] = df_new[col_te].fillna(self.global_mean)

    return df_new.drop(columns=[self.col_name])

# 사용하지 않는 함수입니다.
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

class ExtractNumber(BaseEncoder):
  def __init__(self, column = None):
    """
    Parameters:
      column (str): 숫자를 추출할 컬럼명
    Return:
      column_Number (Float)
    """
    if column is None:
      column = ["Episode_Title"]
    self.column = column

  def fit(self, df):
    return df

  def transform(self, df):
    df = df.copy()
    for col in self.column:
      new_col = f"{col}_Number"
      df[new_col] = df[col].apply(lambda x: float(re.search(r'\d+', str(x)).group()) if re.search(r'\d+', str(x)) else None)
      df.drop(columns=[col], inplace=True)
    return df


class OutlierRemover(BasePreprocessor):
  def __init__(self, columns=None, method="IQR", threshold=1.5):
    """
    Parameters:
        columns (list): 이상치를 제거할 열 리스트
        method (str): 'IQR' 또는 'zscore'
        threshold (float): IQR multiplier or z-score threshold
    """
    self.columns = columns
    self.method = method
    self.threshold = threshold

  def fit(self, df):
    return self

  def transform(self, df):
    df = df.copy()
    for col in self.columns:
      if self.method == "IQR":
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - self.threshold * IQR
        upper_bound = Q3 + self.threshold * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
      elif self.method == "zscore":
        mean = df[col].mean()
        std = df[col].std()
        z = (df[col] - mean) / std
        df = df[(z.abs() <= self.threshold)]
      else:
        raise ValueError("지원하지 않는 이상치 제거 방법입니다.")
    return df

class PercentageOutlierRemover(BasePreprocessor):
  def __init__(self, columns=None, low_bound=0, high_bound=100):
    self.columns = columns
    self.low_bound = low_bound
    self.high_bound = high_bound

  def fit(self, df):
    return self

  def transform(self, df):
    df = df.copy()
    for col in self.columns:
      df = df[(df[col] >= self.low_bound) & (df[col] <= self.high_bound)]
    return df