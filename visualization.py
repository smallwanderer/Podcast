import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def outlier_check(df: pd.DataFrame):
  # 필요한 수치형 컬럼만 선택
  df = df.drop(columns=["Listening_Time_minutes", "id"], errors='ignore')
  df_numeric = df.select_dtypes(include=[int, float])

  # Unnamed: 0 같은 불필요한 컬럼 제거
  if 'Unnamed: 0' in df_numeric.columns:
    df_numeric = df_numeric.drop(columns=['Unnamed: 0'])

  # 대용량 데이터 처리: 75만 개 전부 그리면 너무 무거우니까,
  # 랜덤 샘플링해서 일부만 시각화 (예: 1만개 샘플)
  # sample_size = 10000
  # if len(df_numeric) > sample_size:
  #   df_numeric = df_numeric.sample(sample_size, random_state=42)

  # Scatter Plot 그리기
  for col in df_numeric.columns:
    plt.figure(figsize=(8, 4))
    plt.scatter(np.arange(len(df_numeric)), df_numeric[col], alpha=0.5, s=10)
    plt.title(f'Scatter Plot of {col}')
    plt.xlabel('Sample Index')
    plt.ylabel(col)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
  dataset = "./datasets/preprocessed03/preprocessed03.csv"
  df = pd.read_csv(dataset)

  outlier_check(df)