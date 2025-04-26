from preprocessing import *
import os
import pandas as pd
import yaml
import shutil
from ydata_profiling import ProfileReport

name = 'preprocessed03'
dataset_dir = './datasets'

print(f"Making File Name {name} ... ")
if not os.path.isdir(f"{dataset_dir}/{name}"):
  os.makedirs(f"{dataset_dir}/{name}")

print(f"Copying Config File to {dataset_dir}/{name}")
if not os.path.isfile(f"{dataset_dir}/{name}/{name}_config.yaml"):
  shutil.copy("./preprocessing_config.yaml", f"{dataset_dir}/{name}")
  os.rename(f'{dataset_dir}/{name}/preprocessing_config.yaml', f'{dataset_dir}/{name}/{name}_config.yaml')

preprocessor_classes = {
    "GroupMeanPreprocessor": GroupMeanPreprocessor,
    "FillPreprocessor": FillPreprocessor,
    "LabelEncoding": LabelEncoding,
    "OneHotEncoding": OneHotEncoding,
    "TargetEncodingKFold": TargetEncodingKFold,
    "DictMapping": DictMapping,
    "ExtractNumber" : ExtractNumber,
    "OutlierRemover" : OutlierRemover,
    "PercentageOutlierRemover": PercentageOutlierRemover,
}

class PreprocessingPipeline:
  def __init__(self, steps):
    """
    Parameters:
        steps (list): (name, transformer)
    """
    self.steps = steps

  def fit(self, df):
    for name, transformer in self.steps:
      print(f"Fitting: {name}")
      transformer.fit(df)
    return self

  def transform(self, df):
    for name, transformer in self.steps:
      print(f"Transforming: {name}")
      df = transformer.transform(df)
    return df

  def fit_transform(self, df):
    self.fit(df)
    return self.transform(df)

class PreprocessingPipelineFromConfig:
  def __init__(self, config_path):
    self.config_path = config_path
    self.steps = []

  def load_config(self):
    with open(self.config_path, 'r') as file:
      config = yaml.safe_load(file)

    for step_cfg in config['preprocessing']:
      name = step_cfg['name']
      cls_name = step_cfg['class']
      params = step_cfg.get('params', {})

      transformer_class = preprocessor_classes[cls_name]
      transformer = transformer_class(**params)

      self.steps.append((name, transformer))

    return PreprocessingPipeline(self.steps)

train = './datasets/train.csv'
test = './datasets/test.csv'
train_df = pd.read_csv(train)
test_df = pd.read_csv(test)

config_path = f'{dataset_dir}/{name}/{name}_config.yaml'
builder = PreprocessingPipelineFromConfig(config_path)
pipeline = builder.load_config()

train_df = pipeline.fit(train_df).transform(train_df)
pd.set_option('display.max_columns', None)
print(train_df.head())

train_df.to_csv(f"{dataset_dir}/{name}/{name}.csv")
profile = ProfileReport(train_df, title="Train Data EDA Report", explorative=True)
profile.to_file(f"{dataset_dir}/{name}/{name}_eda_report.html")
test_df = pipeline.transform(test_df)

"""
# 기능 테스트
print("Train Fill Out 실행")
# 'Guest_Popularity_percentage', 'Episode_Length_minutes'
group_inputer = GroupMeanPreprocessor(target_col=['Guest_Popularity_percentage', 'Episode_Length_minutes'])
group_inputer.fit(train_df)
train_df = group_inputer.transform(train_df)

# 'Number_of_Ads'
fill_inputer = FillPreprocessor(method='median', target=['Number_of_Ads'])
fill_inputer.fit(train_df)
train_df = fill_inputer.transform(train_df)

print("Train Encoding 실행")
# 'Podcast_Name', 'Publication_Day', 'Publication_Time'
label_encoder = LabelEncoding(nominal_col=['Podcast_Name', 'Publication_Day', 'Publication_Time'])
label_encoder.fit(train_df)
train_df = label_encoder.transform(train_df)

# 'Genre'
onehot_encoder = OneHotEncoding('Genre')
onehot_encoder.fit(train_df)
train_df = onehot_encoder.transform(train_df)

# 'Episode_Sentiment'
dict_encoder = DictMapping('Episode_Sentiment', {'Positive': 1, 'Neutral': 0, 'Negative': -1})
train_df = dict_encoder.transform(train_df)

pd.set_option('display.max_columns', None)  # 모든 열 출력
pd.set_option('display.max_rows', 10)  # 최대 행 수도 필요 시 조절
print(train_df.head())
"""