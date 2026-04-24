import pandas as pd
import time

input_file = './train.csv'
output_file = './avazu_train_1M.csv'

start_time = time.time()

try:
    df_sample = pd.read_csv(input_file, nrows=1000000) # # to save RAM, we have read only the first million rows, starting from top
    
    print("\nBeginning of the stream:")
    print(df_sample[['id', 'click', 'hour']].head())
    
    print(f"\nEnd of the sampled data:")
    print(df_sample[['id', 'click', 'hour']].tail())
    
    df_sample.to_csv(output_file, index=False)
    
    end_time = time.time()
    print(f"Execution time: {round(end_time - start_time, 2)} seconds.")

except FileNotFoundError:
    print(f"Error: file '{input_file}' not found.")