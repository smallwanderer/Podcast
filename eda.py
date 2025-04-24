import pandas as pd
from ydata_profiling import ProfileReport

dataset = './datasets/preprocessed01.csv'
pd.set_option('display.max_columns', None)  # 모든 열 출력
pd.set_option('display.width', 1000)        # 한 줄 최대 길이 늘리기
pd.set_option('display.max_rows', 100)      # 최대 행 수도 필요 시 조절
df = pd.read_csv(dataset)

print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
print(df.describe())

profile = ProfileReport(df, title="Train Data EDA Report", explorative=True)
profile.to_file("preprocessed01_eda_report.html")