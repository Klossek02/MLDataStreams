import pandas as pd
import numpy as np
from sklearn.feature_extraction import FeatureHasher
import time
import os


base_dir = os.path.dirname(__file__)

input_file = os.path.join(base_dir, 'avazu_train_1M.csv')
output_file = os.path.join(base_dir, 'data', 'avazu_extended.arff')

os.makedirs(os.path.dirname(output_file), exist_ok=True)

n_features = 100
start_time = time.time()

df = pd.read_csv(input_file, nrows=1_000_000)

hour_str = df['hour'].astype(str)
df['hour_of_day'] = hour_str.str[-2:].astype(int)
df['day_of_month'] = hour_str.str[-4:-2].astype(int)
df['day_of_week'] = pd.to_datetime(hour_str, format='%y%m%d%H').dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['time_of_day'] = pd.cut(
    df['hour_of_day'],
    bins=[-1, 5, 11, 17, 23],
    labels=[0, 1, 2, 3]
).astype(int)

df['site_x_hour'] = df['site_id'].astype(str) + "_h" + df['hour_of_day'].astype(str)
df['app_x_hour'] = df['app_id'].astype(str) + "_h" + df['hour_of_day'].astype(str)
df['device_x_hour'] = df['device_type'].astype(str) + "_h" + df['hour_of_day'].astype(str)

window = 1000

df['site_ctr_rolling'] = df.groupby('site_id')['click'] \
    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

df['app_ctr_rolling'] = df.groupby('app_id')['click'] \
    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

df['site_freq_rolling'] = df.groupby('site_id')['click'] \
    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).count())

df['site_freq_delta'] = df['site_freq_rolling'].diff().fillna(0)

window_cols = ['site_ctr_rolling', 'app_ctr_rolling', 'site_freq_rolling', 'site_freq_delta']
df[window_cols] = df[window_cols].fillna(0)

cat_cols = ['C1', 'banner_pos', 'site_id', 'site_domain', 'site_category',
            'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip',
            'device_model', 'device_type', 'device_conn_type',
            'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21',
            'site_x_hour', 'app_x_hour', 'device_x_hour']

X_cat_dict = df[cat_cols].astype(str).to_dict(orient='records')
fh = FeatureHasher(n_features=n_features, input_type='dict')
X_cat_hashed = fh.fit_transform(X_cat_dict).toarray()

time_cols = ['hour_of_day', 'day_of_week', 'is_weekend', 'time_of_day', 'day_of_month']
num_cols = time_cols + window_cols

X_num = df[num_cols].values
X_combined = np.hstack((X_num, X_cat_hashed))
y = df['click'].values

with open(output_file, 'w') as f:
    f.write("@relation avazu_extended_ctr\n\n")
    for col in num_cols:
        f.write(f"@attribute {col} numeric\n")
    for i in range(n_features):
        f.write(f"@attribute hashed_{i} numeric\n")
    f.write("@attribute class {0,1}\n\n")
    f.write("@data\n")
    for i in range(len(y)):
        row_str = ",".join(map(str, X_combined[i]))
        f.write(f"{row_str},{y[i]}\n")

end_time = time.time()

n_rows = X_combined.shape[0]
n_cols = X_combined.shape[1] + 1

print(f"Saved file: {output_file}")
print(f"Execution time: {round(end_time - start_time, 2)} seconds.")
print(f"Rows: {n_rows}")
print(f"Columns (with class): {n_cols}")
print("\nMean of time features:")
print(df[time_cols].mean())
print(f"\nMean site_ctr_rolling: {df['site_ctr_rolling'].mean():.4f}")
print(f"Mean app_ctr_rolling: {df['app_ctr_rolling'].mean():.4f}")
print("\nClass distribution:")
print(df['click'].value_counts(normalize=True))
