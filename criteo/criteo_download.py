import pandas as pd
import numpy as np
from sklearn.feature_extraction import FeatureHasher
import time
import os

base_dir = os.path.dirname(__file__)

input_file = os.path.join(base_dir, 'train.txt')
output_file = os.path.join(base_dir, 'data', 'criteo_100.arff')

os.makedirs(os.path.dirname(output_file), exist_ok=True)

col_names = ['label'] + [f'I{i}' for i in range(1, 14)] + [f'C{i}' for i in range(1, 27)]

n_hashed_features = 100
start_time = time.time()

try:
    df = pd.read_csv(input_file, sep='\t', names=col_names, nrows=1000000)

    num_cols = [f'I{i}' for i in range(1, 14)]
    cat_cols = [f'C{i}' for i in range(1, 27)]

    df[num_cols] = df[num_cols].fillna(0)
    df[cat_cols] = df[cat_cols].fillna('empty')

    X_cat_dict = df[cat_cols].astype(str).to_dict(orient='records')

    fh = FeatureHasher(n_features=n_hashed_features, input_type='dict')
    X_cat_hashed = fh.fit_transform(X_cat_dict).toarray()

    X_num = df[num_cols].values
    y = df['label'].values

    X_combined = np.hstack((X_num, X_cat_hashed))

    with open(output_file, 'w') as f:
        f.write("@relation criteo_mixed_ctr\n\n")

        for i in range(1, 14):
            f.write(f"@attribute num_I{i} numeric\n")

        for i in range(n_hashed_features):
            f.write(f"@attribute hashed_C{i} numeric\n")

        f.write("@attribute class {0,1}\n\n")
        f.write("@data\n")

        for i in range(len(y)):
            row_str = ",".join(map(str, X_combined[i]))
            f.write(f"{row_str},{y[i]}\n")

    end_time = time.time()
    print(f"Saved file: {output_file}")
    print(f"Execution time: {round(end_time - start_time, 2)} seconds.")

except FileNotFoundError:
    print(f"Error: file '{input_file}' not found.")