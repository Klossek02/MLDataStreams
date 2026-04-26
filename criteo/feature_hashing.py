import pandas as pd
import numpy as np
from sklearn.feature_extraction import FeatureHasher
import time
import os

base_dir = os.path.dirname(__file__)

input_file = os.path.join(base_dir, 'train.txt')
output_file = os.path.join(base_dir, 'data', 'criteo_extended.arff')

os.makedirs(os.path.dirname(output_file), exist_ok=True)

n_features = 100
window = 1000
start_time = time.time()

col_names = ['label'] + [f'I{i}' for i in range(1, 14)] + [f'C{i}' for i in range(1, 27)]
df = pd.read_csv(input_file, sep='\t', names=col_names, nrows=1_000_000)

num_cols = [f'I{i}' for i in range(1, 14)]
cat_cols = [f'C{i}' for i in range(1, 27)]

df[num_cols] = df[num_cols].fillna(0)
df[cat_cols] = df[cat_cols].fillna('empty')

rolling_cols = ['I1', 'I2', 'I3', 'I4', 'I5']

for col in rolling_cols:
    df[f'{col}_rolling_mean'] = df[col].shift(1).rolling(window, min_periods=1).mean()
    df[f'{col}_rolling_std'] = df[col].shift(1).rolling(window, min_periods=1).std()

for col in rolling_cols:
    df[f'{col}_delta'] = df[col] - df[f'{col}_rolling_mean']

window_num_cols = []
for col in rolling_cols:
    window_num_cols += [f'{col}_rolling_mean', f'{col}_rolling_std', f'{col}_delta']

df[window_num_cols] = df[window_num_cols].fillna(0)

X_cat_dict = df[cat_cols].astype(str).to_dict(orient='records')
fh = FeatureHasher(n_features=n_features, input_type='dict')
X_cat_hashed = fh.fit_transform(X_cat_dict).toarray()

X_num_original = df[num_cols].values
X_num_window = df[window_num_cols].values
X_combined = np.hstack((X_num_original, X_num_window, X_cat_hashed))
y = df['label'].values

with open(output_file, 'w') as f:
    f.write("@relation criteo_extended_ctr\n\n")
    for col in num_cols:
        f.write(f"@attribute num_{col} numeric\n")
    for col in window_num_cols:
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
print("\nMean of original I1-I5:")
print(df[rolling_cols].mean())
print("\nMean of rolling_mean for I1-I5:")
print(df[[f'{c}_rolling_mean' for c in rolling_cols]].mean())
print("\nClass distribution:")
print(df['label'].value_counts(normalize=True))
