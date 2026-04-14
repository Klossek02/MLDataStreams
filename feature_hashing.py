import pandas as pd
from sklearn.feature_extraction import FeatureHasher
import time
import numpy as np

input_file = 'avazu_train_1M.csv'
output_file = 'avazu_hashed_100.arff'


n_features = 100  # how many columns do we want to compress our data to? - 100 seems to be a good balance

start_time = time.time()
df = pd.read_csv(input_file)

y = df['click'].values
X_raw = df.drop(columns=['id', 'click'])

X_dict = X_raw.astype(str).to_dict(orient='records')


fh = FeatureHasher(n_features=n_features, input_type='dict')
X_hashed = fh.fit_transform(X_dict)


X_dense = X_hashed.toarray()


with open(output_file, 'w') as f:
    # arff file header
    f.write("@relation avazu_hashed_ctr\n\n")
    
    # defining 100 numeric features
    for i in range(n_features):
        f.write(f"@attribute feature_{i} numeric\n")
        
    # target class at the end and defined as a nominal category
    f.write("@attribute class {0,1}\n\n")
    f.write("@data\n")
    
    for i in range(len(y)):
        row_str = ",".join(map(str, X_dense[i]))
        f.write(f"{row_str},{y[i]}\n")

end_time = time.time()
print(f"Execution time: {round(end_time - start_time, 2)} seconds.")