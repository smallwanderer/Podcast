import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# CSV 파일 로딩
df = pd.read_csv("./datasets/preprocessed01.csv")

# 타겟 컬럼 제거
df = df.drop(columns=["Listening_Time_minutes", "log_target"], errors='ignore')

# 🔄 bool (True/False) → int (1/0) 변환
df = df.applymap(lambda x: int(x) if isinstance(x, bool) else x)

# 🔁 Genre 중 하나 제거
genre_columns = sorted([col for col in df.columns if col.startswith("Genre_")])
if genre_columns:
    df = df.drop(columns=[genre_columns[0]])
    print(f"✅ Removed column: {genre_columns[0]}")

# ✅ 반드시 여기서 수치형 컬럼 선택!
df_numeric = df.select_dtypes(include=[int, float])

# 결측치 처리
df_numeric = df_numeric.fillna(df_numeric.mean())

# 상수항 추가
X = sm.add_constant(df_numeric)

# VIF 계산
vif_df = pd.DataFrame()
vif_df["Feature"] = X.columns
vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# 결과 출력
vif_df = vif_df.sort_values(by="VIF", ascending=False)
print(vif_df)
