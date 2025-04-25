import preprocessing
import pandas as pd

train = './datasets/train.csv'
test = './datasets/test.csv'
train_df = pd.read_csv(train)
test_df = pd.read_csv(test)

print("Train Fill Out 실행")
# 'Guest_Popularity_percentage', 'Episode_Length_minutes'
group_inputer = preprocessing.GroupMeanPreprocessor()
group_inputer.fit(train_df)
train_df = group_inputer.transform(train_df)

# 'Number_of_Ads'
fill_inputer = preprocessing.FillPreprocessor(method='median')
fill_inputer.fit(train_df)
train_df = fill_inputer.transform(train_df)

print("Train Encoding 실행")
# 'Podcast_Name', 'Publication_Day', 'Publication_Time'
label_encoder = preprocessing.LabelEncoding()
label_encoder.fit(train_df)
train_df = label_encoder.transform(train_df)

# 'Genre'
onehot_encoder = preprocessing.OneHotEncoding('Genre')
onehot_encoder.fit(train_df)
train_df = onehot_encoder.transform(train_df)

# 'Episode_Sentiment'
dict_encoder = preprocessing.DictMapping('Episode_Sentiment', {'Positive': 1, 'Neutral': 0, 'Negative': -1})
train_df = dict_encoder.transform(train_df)

pd.set_option('display.max_columns', None)  # 모든 열 출력
pd.set_option('display.max_rows', 10)  # 최대 행 수도 필요 시 조절
print(train_df.head())





